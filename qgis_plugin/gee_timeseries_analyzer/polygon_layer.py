"""The QGIS side of the polygons: a memory layer ("GEE Analysis Polygons")
holding every polygon that will be sent to Earth Engine, the map tools that
fill it (free-hand polygon drawing -- the Code Editor's drawing tools -- and
click-a-field picking -- the Field Inspector's Map.onClick), importing from
any polygon layer / shapefile, and turning a GEE table asset into a real
QGIS vector layer styled by crop type."""
import json
import os
import re

from qgis.core import (
    QgsProject, QgsVectorLayer, QgsField, QgsFeature, QgsGeometry, QgsPointXY,
    QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsFeatureRequest,
    QgsRectangle, QgsWkbTypes, QgsFillSymbol, QgsSymbolLayer, QgsProperty,
    QgsPalLayerSettings, QgsVectorLayerSimpleLabeling, QgsTextFormat,
    QgsTextBufferSettings, QgsExpression, QgsExpressionContext,
    QgsExpressionContextUtils, QgsCategorizedSymbolRenderer, QgsRendererCategory,
    QgsApplication,
)
from qgis.gui import QgsMapTool, QgsRubberBand
from qgis.PyQt.QtCore import Qt, QVariant, QPoint, pyqtSignal
from qgis.PyQt.QtGui import QColor

from .gee_client import POLYGON_COLOURS

ANALYSIS_LAYER_NAME = "GEE Analysis Polygons"
ANALYSIS_LAYER_PROP = "gee_timeseries_analyzer/analysis_layer"
WGS84 = QgsCoordinateReferenceSystem("EPSG:4326")

# Crop palette of the Field Inspector app; anything else gets a generated colour.
CROP_COLOURS = {
    "winter-wheat":       "#f5d76e",
    "wheat-rice":         "#ab964d",
    "cotton":             "#f0f0f0",
    "alfalfa":            "#2ecc71",
    "orchard":            "#27ae60",
    "orchard/winter":     "#1a7a4a",
    "orchard/alfalfa":    "#52d68a",
    "orchard/vegetables": "#82e0aa",
    "orchard/wheat":      "#a9cfa4",
    "vineyard":           "#8e44ad",
    "greenhouse":         "#e8d5b7",
}
NA_COLOUR = "#aaaaaa"


# ---------------------------------------------------------------------------
# Analysis polygon layer
# ---------------------------------------------------------------------------
def _new_analysis_layer():
    layer = QgsVectorLayer("MultiPolygon?crs=EPSG:4326", ANALYSIS_LAYER_NAME, "memory")
    layer.dataProvider().addAttributes([
        QgsField("pid", QVariant.Int),
        QgsField("name", QVariant.String),
        QgsField("color", QVariant.String),
        QgsField("source", QVariant.String),
        QgsField("src_key", QVariant.String),   # "<layer id>:<feature id>" of an imported/picked feature
    ])
    layer.updateFields()
    layer.setCustomProperty(ANALYSIS_LAYER_PROP, True)
    _style_analysis_layer(layer)
    return layer


def _style_analysis_layer(layer):
    symbol = QgsFillSymbol.createSimple({
        "color": "255,107,107,80", "outline_color": "255,107,107", "outline_width": "0.8",
    })
    sl = symbol.symbolLayer(0)
    sl.setDataDefinedProperty(QgsSymbolLayer.PropertyFillColor,
                              QgsProperty.fromExpression("set_color_part(\"color\", 'alpha', 70)"))
    sl.setDataDefinedProperty(QgsSymbolLayer.PropertyStrokeColor, QgsProperty.fromField("color"))
    layer.renderer().setSymbol(symbol)

    settings = QgsPalLayerSettings()
    settings.fieldName = "name"
    fmt = QgsTextFormat()
    fmt.setSize(9)
    buf = QgsTextBufferSettings()
    buf.setEnabled(True)
    buf.setSize(0.8)
    buf.setColor(QColor("white"))
    fmt.setBuffer(buf)
    settings.setFormat(fmt)
    layer.setLabeling(QgsVectorLayerSimpleLabeling(settings))
    layer.setLabelsEnabled(True)
    layer.triggerRepaint()


def find_analysis_layer():
    for lyr in QgsProject.instance().mapLayers().values():
        if lyr.customProperty(ANALYSIS_LAYER_PROP) and isinstance(lyr, QgsVectorLayer):
            return lyr
    return None


def get_or_create_analysis_layer():
    layer = find_analysis_layer()
    if layer is None:
        layer = _new_analysis_layer()
        QgsProject.instance().addMapLayer(layer)
    return layer


def next_pid(layer):
    pids = [f["pid"] for f in layer.getFeatures() if f["pid"] is not None]
    return (max(pids) + 1) if pids else 1


def add_polygon(layer, geom_4326, name=None, source="drawn", src_key=None):
    """Adds one polygon (already in EPSG:4326). Returns its pid."""
    pid = next_pid(layer)
    colour = POLYGON_COLOURS[(layer.featureCount()) % len(POLYGON_COLOURS)]
    geom = QgsGeometry(geom_4326)
    if not geom.isGeosValid():
        geom = geom.makeValid()
    geom.convertToMultiType()
    feat = QgsFeature(layer.fields())
    feat.setGeometry(geom)
    values = {"pid": pid, "name": name or f"Polygon {pid}", "color": colour,
              "source": source, "src_key": src_key}
    for field, value in values.items():
        if layer.fields().indexOf(field) >= 0:   # older layers lack src_key
            feat[field] = value
    layer.dataProvider().addFeatures([feat])
    layer.updateExtents()
    layer.triggerRepaint()
    return pid


def clear_polygons(layer):
    ids = [f.id() for f in layer.getFeatures()]
    layer.dataProvider().deleteFeatures(ids)
    layer.triggerRepaint()


def remove_polygons(layer, pids):
    ids = [f.id() for f in layer.getFeatures() if f["pid"] in pids]
    layer.dataProvider().deleteFeatures(ids)
    layer.triggerRepaint()


def read_polygons(layer):
    """-> list of {pid, name, color, geojson} (geometry in EPSG:4326)."""
    out = []
    if layer is None:
        return out
    for f in layer.getFeatures(QgsFeatureRequest().addOrderBy("pid")):
        g = f.geometry()
        if g is None or g.isEmpty():
            continue
        out.append({
            "pid": int(f["pid"]),
            "name": f["name"] or f"Polygon {f['pid']}",
            "color": f["color"] or POLYGON_COLOURS[0],
            "geojson": json.loads(g.asJson(7)),
            "bbox": g.boundingBox(),
            "fid": f.id(),
        })
    return out


def source_key(src_layer, feature):
    return f"{src_layer.id()}:{feature.id()}"


def find_pid_by_source(layer, key):
    if layer is None or layer.fields().indexOf("src_key") < 0:
        return None
    for f in layer.getFeatures():
        if f["src_key"] == key:
            return int(f["pid"])
    return None


def feature_ids_for_pids(layer, pids):
    return [f.id() for f in layer.getFeatures() if f["pid"] in pids]


def to_wgs84(geom, src_crs):
    g = QgsGeometry(geom)
    if src_crs != WGS84:
        g.transform(QgsCoordinateTransform(src_crs, WGS84, QgsProject.instance()))
    return g


def label_for(layer, feature, expression):
    """Evaluates a QGIS expression (e.g. 'Field ' || "fid") for a feature;
    falls back to the feature id."""
    if expression:
        exp = QgsExpression(expression)
        ctx = QgsExpressionContext(QgsExpressionContextUtils.globalProjectLayerScopes(layer))
        ctx.setFeature(feature)
        val = exp.evaluate(ctx)
        if not exp.hasEvalError() and val not in (None, ""):
            return str(val)
    return f"{layer.name()} #{feature.id()}"


def import_features(analysis_layer, src_layer, features, name_expression, source="imported"):
    """Copies polygon features from any layer into the analysis layer.
    Returns the list of new pids."""
    pids = []
    for feat in features:
        g = feat.geometry()
        if g is None or g.isEmpty() or g.type() != QgsWkbTypes.PolygonGeometry:
            continue
        g = to_wgs84(g, src_layer.crs())
        pids.append(add_polygon(analysis_layer, g, label_for(src_layer, feat, name_expression), source,
                                source_key(src_layer, feat)))
    return pids


def layer_union_geojson(layer, selected_only=False, max_vertices=50000):
    """Dissolved (selected) polygons of a layer in EPSG:4326 as GeoJSON,
    simplified in ~5 m steps if it has more than `max_vertices` vertices
    (keeps Earth Engine requests small). Raises ValueError if empty."""
    feats = layer.getSelectedFeatures() if selected_only else layer.getFeatures()
    geoms = [f.geometry() for f in feats if f.geometry() is not None and not f.geometry().isEmpty()]
    if not geoms:
        raise ValueError(f"'{layer.name()}' has no (selected) polygons.")
    g = to_wgs84(QgsGeometry.unaryUnion(geoms), layer.crs())
    if not g.isGeosValid():
        g = g.makeValid()
    tol = 0.00005
    while g.constGet().nCoordinates() > max_vertices and tol < 0.01:
        g = g.simplify(tol)
        tol *= 2
    return json.loads(g.asJson(7))


def is_polygon_layer(layer):
    return isinstance(layer, QgsVectorLayer) and layer.geometryType() == QgsWkbTypes.PolygonGeometry


def extent_in_canvas_crs(layer, canvas):
    return canvas.mapSettings().layerExtentToOutputExtent(layer, layer.extent())


# ---------------------------------------------------------------------------
# GEE table asset -> GeoJSON file -> QGIS layer
# ---------------------------------------------------------------------------
def _asset_cache_dir():
    d = os.path.join(QgsApplication.qgisSettingsDirPath(), "gee_timeseries_analyzer", "assets")
    os.makedirs(d, exist_ok=True)
    return d


def add_asset_layer(asset_id, features, class_field=None):
    """Writes the downloaded features to a GeoJSON file in the QGIS profile
    folder (so the layer survives saving/reopening the project) and adds it
    to the project, replacing a previous copy of the same asset."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", asset_id)
    path = os.path.join(_asset_cache_dir(), safe + ".geojson")
    for f in features:
        f.pop("id", None)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": features}, fh)

    name = asset_id.split("/")[-1]
    project = QgsProject.instance()
    for lyr in project.mapLayersByName(name):
        if lyr.source().split("|")[0] == path:
            project.removeMapLayer(lyr.id())
    layer = QgsVectorLayer(path, name, "ogr")
    if not layer.isValid():
        raise RuntimeError(f"Could not open downloaded asset {path}")
    layer.setCustomProperty("gee_timeseries_analyzer/asset_id", asset_id)
    if class_field and layer.fields().indexOf(class_field) >= 0:
        style_by_class(layer, class_field)
    project.addMapLayer(layer)
    return layer


def _generated_colour(i):
    return QColor.fromHsv((i * 47) % 360, 160, 220).name()


def style_by_class(layer, field):
    """Categorised fill, Field-Inspector palette for known crops, translucent
    fill (the original used colour + '99' alpha)."""
    values = sorted({str(v) for v in layer.uniqueValues(layer.fields().indexOf(field))
                     if v is not None and str(v) not in ("", "NULL")})
    categories = []
    for i, v in enumerate(values):
        hexc = CROP_COLOURS.get(v, _generated_colour(i))
        categories.append(QgsRendererCategory(v, _class_symbol(hexc), v))
    # an empty value is the categorised renderer's "all other values" class
    categories.append(QgsRendererCategory("", _class_symbol(NA_COLOUR), "NA / unknown"))
    layer.setRenderer(QgsCategorizedSymbolRenderer(field, categories))
    layer.triggerRepaint()


def _class_symbol(hexc):
    c = QColor(hexc)
    return QgsFillSymbol.createSimple({
        "color": f"{c.red()},{c.green()},{c.blue()},153",
        "outline_color": f"{c.red()},{c.green()},{c.blue()}",
        "outline_width": "0.3",
    })


# ---------------------------------------------------------------------------
# Map tools
# ---------------------------------------------------------------------------
class PolygonDrawTool(QgsMapTool):
    """Free-hand polygon digitising on the canvas. Left click adds a vertex,
    right click / double click / Enter finishes (needs >= 3 vertices),
    Backspace removes the last vertex, Esc discards. Emits the polygon in
    the canvas CRS."""
    polygonDrawn = pyqtSignal(QgsGeometry)

    def __init__(self, canvas):
        super().__init__(canvas)
        self.points = []
        self.rubber = QgsRubberBand(canvas, QgsWkbTypes.PolygonGeometry)
        self.set_colour(POLYGON_COLOURS[0])
        self.setCursor(Qt.CrossCursor)

    def set_colour(self, hexc):
        c = QColor(hexc)
        self.rubber.setStrokeColor(c)
        fill = QColor(c)
        fill.setAlpha(60)
        self.rubber.setFillColor(fill)
        self.rubber.setWidth(2)

    def _redraw(self, cursor=None):
        self.rubber.reset(QgsWkbTypes.PolygonGeometry)
        pts = self.points + ([cursor] if cursor is not None else [])
        for i, p in enumerate(pts):
            self.rubber.addPoint(p, i == len(pts) - 1)

    def canvasMoveEvent(self, event):
        if self.points:
            self._redraw(self.toMapCoordinates(event.pos()))

    def canvasReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.points.append(QgsPointXY(self.toMapCoordinates(event.pos())))
            self._redraw()
        elif event.button() == Qt.RightButton:
            self.finish()

    def canvasDoubleClickEvent(self, event):
        # the double click's release already appended the vertex
        self.finish()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.reset()
        elif event.key() == Qt.Key_Backspace and self.points:
            self.points.pop()
            self._redraw()
        elif event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.finish()

    def finish(self):
        pts = []
        for p in self.points:   # drop consecutive duplicates (double click)
            if not pts or p != pts[-1]:
                pts.append(p)
        if len(pts) >= 3:
            geom = QgsGeometry.fromPolygonXY([pts + [pts[0]]])
            self.reset()
            self.polygonDrawn.emit(geom)
        else:
            self.reset()

    def reset(self):
        self.points = []
        self.rubber.reset(QgsWkbTypes.PolygonGeometry)

    def deactivate(self):
        self.reset()
        super().deactivate()


class FeaturePickTool(QgsMapTool):
    """Click-a-field map tool: emits featurePicked(QgsFeature) for the
    polygon of `layer_getter()` under the click (the Field Inspector's
    Map.onClick + filterBounds(point)). `layer_getter` is a callable so the
    tool follows whichever layer is currently selected in the dock."""
    featurePicked = pyqtSignal(object)
    nothingPicked = pyqtSignal()

    TOL_PX = 3

    def __init__(self, canvas, layer_getter):
        super().__init__(canvas)
        self._layer_getter = layer_getter
        self.setCursor(Qt.PointingHandCursor)

    def canvasReleaseEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        layer = self._layer_getter()
        if layer is None:
            return
        ms = self.canvas().mapSettings()
        click = ms.mapToLayerCoordinates(layer, self.toMapCoordinates(event.pos()))
        p1 = self.toMapCoordinates(event.pos() - QPoint(self.TOL_PX, self.TOL_PX))
        p2 = self.toMapCoordinates(event.pos() + QPoint(self.TOL_PX, self.TOL_PX))
        rect = ms.mapToLayerCoordinates(layer, QgsRectangle(p1, p2))
        click_geom = QgsGeometry.fromPointXY(click)

        candidates = list(layer.getFeatures(QgsFeatureRequest(rect)))
        for feat in candidates:
            if feat.geometry().contains(click_geom):
                self.featurePicked.emit(feat)
                return
        if candidates:   # click on an edge, within tolerance
            self.featurePicked.emit(candidates[0])
        else:
            self.nothingPicked.emit()
