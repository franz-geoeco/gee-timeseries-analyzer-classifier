"""Exposes the GEE Time Series Analyzer's extraction as QGIS Processing
algorithms, so it's usable from Model Builder, batch mode and `qgis_process`
-- e.g. to extract a whole season for hundreds of fields without the dock."""
import json
import os

from qgis.core import (
    QgsProcessing, QgsProcessingAlgorithm, QgsProcessingProvider, QgsProcessingException,
    QgsProcessingParameterFeatureSource, QgsProcessingParameterExpression,
    QgsProcessingParameterString, QgsProcessingParameterEnum, QgsProcessingParameterNumber,
    QgsProcessingParameterFeatureSink, QgsProcessingParameterBoolean, QgsProcessingParameterFile,
    QgsProcessingParameterRasterLayer, QgsProcessingParameterRasterDestination,
    QgsFields, QgsField, QgsFeature, QgsFeatureSink, QgsWkbTypes, QgsGeometry,
    QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsExpression, QgsJsonUtils,
)
from qgis.PyQt.QtCore import QVariant, QDate
from qgis.PyQt.QtGui import QIcon

from . import gee_client as c

ICON_PATH = os.path.join(os.path.dirname(__file__), "icons", "gee_icon.svg")
PRODUCT_KEYS = list(c.PRODUCTS.keys())
PRODUCT_LABELS = [c.PRODUCTS[k]["label"] for k in PRODUCT_KEYS]
INDEX_KEYS = list(c.OPTICAL_INDICES.keys())
WGS84 = QgsCoordinateReferenceSystem("EPSG:4326")


def _progress_feedback(feedback):
    def progress(i, n, msg):
        feedback.setProgress(int(100 * i / max(n, 1)))
        feedback.pushInfo(msg)
    return progress


class GEEProcessingProvider(QgsProcessingProvider):
    def id(self):
        return "geetimeseriesanalyzer"

    def name(self):
        return "GEE Time Series Analyzer & Classifier"

    def icon(self):
        return QIcon(ICON_PATH)

    def loadAlgorithms(self):
        self.addAlgorithm(ExtractTimeSeriesAlgorithm())
        self.addAlgorithm(DownloadAssetAlgorithm())
        self.addAlgorithm(ApplyModelAlgorithm())
        self.addAlgorithm(FieldMajorityAlgorithm())
        self.addAlgorithm(TransitionMatrixAlgorithm())


class _Base(QgsProcessingAlgorithm):
    def icon(self):
        return QIcon(ICON_PATH)

    def createInstance(self):
        return type(self)()

    def _init_ee(self, feedback):
        try:
            c.ensure_initialised()
        except Exception as e:
            raise QgsProcessingException(
                f"Earth Engine is not available/initialised: {e}. Sign in via the "
                "Google Earth Engine plugin first.")


class ExtractTimeSeriesAlgorithm(_Base):
    INPUT = "INPUT"
    NAME = "NAME"
    START = "START"
    END = "END"
    PRODUCTS = "PRODUCTS"
    INDEX = "INDEX"
    MAX_CLOUD = "MAX_CLOUD"
    MIN_VALID = "MIN_VALID"
    MERGE_DAYS = "MERGE_DAYS"
    OUTPUT = "OUTPUT"

    def name(self):
        return "extract_polygon_time_series"

    def displayName(self):
        return "Extract polygon time series (S1 / S2 / Landsat)"

    def shortHelpString(self):
        return ("Mean Sentinel-1 VV/VH (descending + ascending, dB), Sentinel-2 and Landsat 8/9 "
                "optical index per polygon and acquisition date, computed in Google Earth Engine. "
                "Output is a table: polygon, pid, series, date, value, valid_fraction. Rows where "
                "less than 'Min. valid pixels' of the polygon is usable (clouds / outside scene / "
                "S1 border noise) are dropped.")

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.INPUT, "Polygons", [QgsProcessing.TypeVectorPolygon]))
        self.addParameter(QgsProcessingParameterExpression(
            self.NAME, "Polygon name", defaultValue="$id", parentLayerParameterName=self.INPUT))
        year = QDate.currentDate().year()
        self.addParameter(QgsProcessingParameterString(self.START, "Start date (YYYY-MM-DD)",
                                                       defaultValue=f"{year}-01-01"))
        self.addParameter(QgsProcessingParameterString(self.END, "End date, exclusive (YYYY-MM-DD)",
                                                       defaultValue=f"{year + 1}-01-01"))
        self.addParameter(QgsProcessingParameterEnum(
            self.PRODUCTS, "Products", options=PRODUCT_LABELS, allowMultiple=True,
            defaultValue=list(range(len(PRODUCT_KEYS)))))
        self.addParameter(QgsProcessingParameterEnum(
            self.INDEX, "Optical index", options=INDEX_KEYS, defaultValue=0))
        self.addParameter(QgsProcessingParameterNumber(
            self.MAX_CLOUD, "Max. scene cloud cover (%)", QgsProcessingParameterNumber.Integer,
            defaultValue=70, minValue=1, maxValue=100))
        self.addParameter(QgsProcessingParameterNumber(
            self.MIN_VALID, "Min. valid pixels in polygon (%)", QgsProcessingParameterNumber.Integer,
            defaultValue=50, minValue=0, maxValue=100))
        self.addParameter(QgsProcessingParameterBoolean(
            self.MERGE_DAYS, "Merge same-day acquisitions", defaultValue=True))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT, "Time series", QgsProcessing.TypeVector))

    def processAlgorithm(self, parameters, context, feedback):
        self._init_ee(feedback)
        source = self.parameterAsSource(parameters, self.INPUT, context)
        name_expr = QgsExpression(self.parameterAsExpression(parameters, self.NAME, context))
        start = self.parameterAsString(parameters, self.START, context)
        end = self.parameterAsString(parameters, self.END, context)
        products = [PRODUCT_KEYS[i] for i in self.parameterAsEnums(parameters, self.PRODUCTS, context)]
        index = INDEX_KEYS[self.parameterAsEnum(parameters, self.INDEX, context)]
        max_cloud = self.parameterAsInt(parameters, self.MAX_CLOUD, context)
        min_valid = self.parameterAsInt(parameters, self.MIN_VALID, context) / 100.0
        merge = self.parameterAsBool(parameters, self.MERGE_DAYS, context)

        xform = QgsCoordinateTransform(source.sourceCrs(), WGS84, context.transformContext())
        ctx = self.createExpressionContext(parameters, context, source)
        name_expr.prepare(ctx)
        polygons, names = [], {}
        for pid, feat in enumerate(source.getFeatures(), start=1):
            g = feat.geometry()
            if g is None or g.isEmpty():
                continue
            g.transform(xform)
            ctx.setFeature(feat)
            val = name_expr.evaluate(ctx)
            names[pid] = str(val) if val not in (None, "") else f"Polygon {pid}"
            polygons.append({"pid": pid, "geojson": json.loads(g.asJson(7))})
        if not polygons:
            raise QgsProcessingException("No polygons in the input.")
        feedback.pushInfo(f"{len(polygons)} polygon(s), {start} – {end}, products: {', '.join(products)}")

        df, errors = c.extract_time_series(polygons, start, end, products, index, max_cloud,
                                           progress_cb=_progress_feedback(feedback),
                                           cancel_cb=feedback.isCanceled)
        for product, err in errors.items():
            feedback.reportError(f"{c.PRODUCTS[product]['label']}: {err}")
        if merge:
            df = c.filter_series(df, min_valid)
        else:
            df = df[(df["valid"] >= min_valid) & df["value"].notna()]

        fields = QgsFields()
        fields.append(QgsField("polygon", QVariant.String))
        fields.append(QgsField("pid", QVariant.Int))
        fields.append(QgsField("series", QVariant.String))
        fields.append(QgsField("date", QVariant.Date))
        fields.append(QgsField("value", QVariant.Double))
        fields.append(QgsField("valid_fraction", QVariant.Double))
        sink, dest_id = self.parameterAsSink(parameters, self.OUTPUT, context, fields,
                                             QgsWkbTypes.NoGeometry, WGS84)
        for row in df.itertuples(index=False):
            meta = c.SERIES[row.series]
            label = f"{meta['label']} {index}" if meta["unit"] is None else meta["label"]
            f = QgsFeature(fields)
            f.setAttributes([names.get(int(row.pid)), int(row.pid), label,
                             QDate(row.date.year, row.date.month, row.date.day),
                             float(row.value), float(row.valid)])
            sink.addFeature(f, QgsFeatureSink.FastInsert)
        feedback.setProgress(100)
        return {self.OUTPUT: dest_id}


class DownloadAssetAlgorithm(_Base):
    ASSET = "ASSET"
    OUTPUT = "OUTPUT"

    def name(self):
        return "download_table_asset"

    def displayName(self):
        return "Download GEE table asset as vector layer"

    def shortHelpString(self):
        return "Downloads every feature of an Earth Engine FeatureCollection asset (paged, no 5000 limit)."

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterString(
            self.ASSET, "Asset id", defaultValue=""))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, "Features"))

    def processAlgorithm(self, parameters, context, feedback):
        self._init_ee(feedback)
        asset = self.parameterAsString(parameters, self.ASSET, context)
        feats = c.fetch_asset_features(asset, progress_cb=_progress_feedback(feedback))
        if not feats:
            raise QgsProcessingException("The asset contains no features.")
        fc_text = json.dumps({"type": "FeatureCollection", "features": feats})
        fields = QgsJsonUtils.stringToFields(fc_text)
        qfeats = QgsJsonUtils.stringToFeatureList(fc_text, fields)
        wkb = QgsWkbTypes.MultiPolygon
        for f in qfeats:
            if f.hasGeometry():
                wkb = QgsWkbTypes.multiType(f.geometry().wkbType())
                break
        sink, dest_id = self.parameterAsSink(parameters, self.OUTPUT, context, fields, wkb, WGS84)
        for f in qfeats:
            if f.hasGeometry():
                g = f.geometry()
                g.convertToMultiType()
                f.setGeometry(g)
            sink.addFeature(f, QgsFeatureSink.FastInsert)
        return {self.OUTPUT: dest_id}



# ---------------------------------------------------------------------------
# Classification algorithms (batch mode: several years / maps in one go)
# ---------------------------------------------------------------------------
def _feedback_cbs(feedback):
    def progress(i, n, msg):
        feedback.setProgress(int(100 * i / max(n, 1)))
        feedback.pushInfo(msg)
    return progress, feedback.isCanceled


def _union_geojson(source, context):
    xform = QgsCoordinateTransform(source.sourceCrs(), WGS84, context.transformContext())
    geoms = [f.geometry() for f in source.getFeatures() if f.hasGeometry()]
    if not geoms:
        raise QgsProcessingException("The area layer has no polygons.")
    g = QgsGeometry.unaryUnion(geoms)
    g.transform(xform)
    if not g.isGeosValid():
        g = g.makeValid()
    return json.loads(g.asJson(7))


def _polygons(source, context):
    xform = QgsCoordinateTransform(source.sourceCrs(), WGS84, context.transformContext())
    out = []
    for f in source.getFeatures():
        if not f.hasGeometry():
            continue
        g = QgsGeometry(f.geometry())
        g.transform(xform)
        out.append({"geojson": json.loads(g.asJson(7)), "id": f.id()})
    return out


class ApplyModelAlgorithm(_Base):
    MODEL, ENGINE, YEAR, AREA, STACK, MASK, SCALE, OUTPUT = (
        "MODEL", "ENGINE", "YEAR", "AREA", "STACK", "MASK", "SCALE", "OUTPUT")
    ENGINES = ["Earth Engine (rebuild the features for YEAR over AREA)", "Local raster stack (offline)"]

    def name(self):
        return "apply_classification_model"

    def displayName(self):
        return "Apply classification model (year / stack)"

    def shortHelpString(self):
        return ("Applies a model file saved in the Classification section. Earth Engine: the model's features are built "
                "for YEAR over AREA, the classifier is retrained from the stored samples and the map is "
                "downloaded as GeoTIFF (class codes, 255 = NoData). Offline: classifies a local raster stack with "
                "the model's band names (scikit-learn). Use batch mode for several years.")

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFile(self.MODEL, "Model file (.json)", extension="json"))
        self.addParameter(QgsProcessingParameterEnum(self.ENGINE, "Engine", options=self.ENGINES, defaultValue=0))
        self.addParameter(QgsProcessingParameterNumber(
            self.YEAR, "Year (Earth Engine)", QgsProcessingParameterNumber.Integer,
            defaultValue=QDate.currentDate().year(), minValue=2017, maxValue=2100))
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.AREA, "Classification area (Earth Engine)", [QgsProcessing.TypeVectorPolygon], optional=True))
        self.addParameter(QgsProcessingParameterRasterLayer(self.STACK, "Raster stack (offline)", optional=True))
        self.addParameter(QgsProcessingParameterString(
            self.MASK, "ESA WorldCover classes to keep (empty = no mask)", defaultValue="40, 10", optional=True))
        self.addParameter(QgsProcessingParameterNumber(
            self.SCALE, "Resolution (m, Earth Engine)", QgsProcessingParameterNumber.Integer,
            defaultValue=10, minValue=10, maxValue=1000))
        self.addParameter(QgsProcessingParameterRasterDestination(self.OUTPUT, "Classified map"))

    def processAlgorithm(self, parameters, context, feedback):
        from . import classification as k
        from . import local_classifier as lc
        try:
            model = k.load_model(self.parameterAsFile(parameters, self.MODEL, context))
        except Exception as e:
            raise QgsProcessingException(f"Model file: {e}")
        mask_txt = self.parameterAsString(parameters, self.MASK, context) or ""
        try:
            mask = [int(v) for v in mask_txt.replace(";", ",").split(",") if v.strip()] or None
        except ValueError:
            raise QgsProcessingException("WorldCover classes must be numbers, e.g. 40, 10.")
        out = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
        progress, cancel = _feedback_cbs(feedback)
        if self.parameterAsEnum(parameters, self.ENGINE, context) == 1:
            stack = self.parameterAsRasterLayer(parameters, self.STACK, context)
            if stack is None:
                raise QgsProcessingException("Choose the raster stack.")
            try:
                lc.run_local_apply(model, stack.source(), out, mask, None, progress, cancel)
            except Exception as e:
                raise QgsProcessingException(str(e))
            return {self.OUTPUT: out}
        if not model.get("features"):
            raise QgsProcessingException("The model has no feature recipe (trained on a composite stack); "
                                         "use the offline engine with a stack of the same bands.")
        self._init_ee(feedback)
        area = self.parameterAsSource(parameters, self.AREA, context)
        if area is None:
            raise QgsProcessingException("Choose the classification area.")
        year = self.parameterAsInt(parameters, self.YEAR, context)
        aoi = {"type": "geojson", "geojson": _union_geojson(area, context)}
        try:
            res = k.run_apply(model, year, aoi, mask, None, progress, cancel)
            feedback.pushInfo(f"Downloading the {year} map …")
            k.download_map(res["map"], res["region"], res["classes"], out,
                           self.parameterAsInt(parameters, self.SCALE, context), None, progress, cancel)
        except InterruptedError:
            raise QgsProcessingException("Cancelled.")
        except Exception as e:
            raise QgsProcessingException(str(e))
        return {self.OUTPUT: out}


class FieldMajorityAlgorithm(_Base):
    INPUT, FIELDS, OUTPUT = "INPUT", "FIELDS", "OUTPUT"

    def name(self):
        return "classification_to_fields"

    def displayName(self):
        return "Aggregate classified map to fields (majority class)"

    def shortHelpString(self):
        return ("Majority class per polygon of a classified GeoTIFF (pixel centres inside the polygon): class code "
                "and name, its share of the field's pixels (confidence), the runner-up class and the pixel count. "
                "Class names are read from the raster's category names.")

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT, "Classified map"))
        self.addParameter(QgsProcessingParameterFeatureSource(self.FIELDS, "Fields", [QgsProcessing.TypeVectorPolygon]))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, "Fields with classes"))

    def processAlgorithm(self, parameters, context, feedback):
        from . import analysis_tools as at
        raster = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        source = self.parameterAsSource(parameters, self.FIELDS, context)
        path = raster.source().split("|")[0]
        names = {code: q["name"] for code, q in at.raster_classes(path).items()}
        try:
            df = at.field_majority(path, _polygons(source, context), cancel_cb=feedback.isCanceled).set_index("id")
        except InterruptedError:
            raise QgsProcessingException("Cancelled.")
        fields = QgsFields(source.fields())
        for name, typ in (("cls_code", QVariant.Int), ("cls_name", QVariant.String), ("cls_share", QVariant.Double),
                          ("cls2_code", QVariant.Int), ("cls2_name", QVariant.String),
                          ("cls2_share", QVariant.Double), ("n_pixels", QVariant.Int)):
            if fields.indexOf(name) < 0:
                fields.append(QgsField(name, typ))
        sink, dest = self.parameterAsSink(parameters, self.OUTPUT, context, fields,
                                          QgsWkbTypes.multiType(source.wkbType()), source.sourceCrs())
        for f in source.getFeatures():
            nf = QgsFeature(fields)
            if f.hasGeometry():
                g = QgsGeometry(f.geometry())
                g.convertToMultiType()
                nf.setGeometry(g)
            attrs = list(f.attributes()) + [None] * (len(fields) - len(f.attributes()))
            nf.setAttributes(attrs)
            if f.id() in df.index:
                r = df.loc[f.id()]
                code = None if r["class_code"] != r["class_code"] or r["class_code"] is None else int(r["class_code"])
                code2 = None if r["second_code"] != r["second_code"] or r["second_code"] is None else int(r["second_code"])
                nf["cls_code"] = code
                nf["cls_name"] = names.get(code) if code is not None else None
                nf["cls_share"] = None if code is None else float(r["share"])
                nf["cls2_code"] = code2
                nf["cls2_name"] = names.get(code2) if code2 is not None else None
                nf["cls2_share"] = None if code2 is None else float(r["second_share"])
                nf["n_pixels"] = int(r["n_pixels"])
            sink.addFeature(nf, QgsFeatureSink.FastInsert)
        return {self.OUTPUT: dest}


class TransitionMatrixAlgorithm(_Base):
    MAP_FROM, MAP_TO, AREA, OUTPUT = "MAP_FROM", "MAP_TO", "AREA", "OUTPUT"

    def name(self):
        return "crop_transition_matrix"

    def displayName(self):
        return "Crop transition matrix (two classified maps)"

    def shortHelpString(self):
        return ("Area (ha) of every class transition between two classified maps (e.g. 2025 -> 2026), "
                "optionally inside polygons. The second map is read on the grid of the first.")

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.MAP_FROM, "Earlier map"))
        self.addParameter(QgsProcessingParameterRasterLayer(self.MAP_TO, "Later map"))
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.AREA, "Restrict to polygons", [QgsProcessing.TypeVectorPolygon], optional=True))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, "Transitions", QgsProcessing.TypeVector))

    def processAlgorithm(self, parameters, context, feedback):
        from . import analysis_tools as at
        a = self.parameterAsRasterLayer(parameters, self.MAP_FROM, context).source().split("|")[0]
        b = self.parameterAsRasterLayer(parameters, self.MAP_TO, context).source().split("|")[0]
        area = self.parameterAsSource(parameters, self.AREA, context)
        mask = [_union_geojson(area, context)] if area is not None else None
        progress, cancel = _feedback_cbs(feedback)
        try:
            rot = at.rotation([(0, a), (1, b)], mask, progress, cancel)
        except InterruptedError:
            raise QgsProcessingException("Cancelled.")
        tt = at.transition_table(rot, 0, 1)
        fields = QgsFields()
        for name, typ in (("from_code", QVariant.Int), ("from_name", QVariant.String), ("to_code", QVariant.Int),
                          ("to_name", QVariant.String), ("ha", QVariant.Double), ("pct_of_from", QVariant.Double),
                          ("changed", QVariant.Bool)):
            fields.append(QgsField(name, typ))
        sink, dest = self.parameterAsSink(parameters, self.OUTPUT, context, fields, QgsWkbTypes.NoGeometry, WGS84)
        for r in tt.itertuples(index=False):
            f = QgsFeature(fields)
            f.setAttributes([int(r.from_code), r.from_name, int(r.to_code), r.to_name, float(r.ha),
                             None if r.pct_of_from is None else float(r.pct_of_from), r.from_code != r.to_code])
            sink.addFeature(f, QgsFeatureSink.FastInsert)
        return {self.OUTPUT: dest}
