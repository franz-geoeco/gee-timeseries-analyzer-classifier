"""Texts of the hover info buttons (style.InfoButton). GROUPS is keyed by
the group box title; the other constants belong to single settings."""

GROUPS = {
    # -- Spectra -------------------------------------------------------------
    "Date and sensors":
        "For every sample and sensor the image <b>closest to the target date</b> within the search window is "
        "used — anywhere on Earth. Sentinel-2 SR (10 m, from 2017) and Landsat 8/9 C2 L2 (30 m, from 2013) "
        "are searched separately, so their dates can differ; the table shows each date and the difference in "
        "days.",
    "Samples":
        "A <b>point</b> gives the reflectance of the pixel under it; a <b>polygon</b> the mean ± standard "
        "deviation of all its pixels. Add samples by clicking, drawing or from the selected features of a "
        "point or polygon layer (up to 10). Samples are processed one after another in the background; "
        "after changing date or settings press <b>Re-sample all</b>.",
    # -- Time series ---------------------------------------------------------
    "Earth Engine":
        "Connection to Google Earth Engine. The plugin uses the sign-in and Cloud project of the official "
        "<b>Google Earth Engine</b> QGIS plugin. Leave <i>Project</i> empty to use that project, or enter "
        "another Cloud project you have access to. Nothing is sent anywhere except to Earth Engine under "
        "your own account.",
    "Draw polygons":
        "Draw any number of polygons on the map. Each one becomes an analysis polygon with its own colour "
        "in the temporary layer <i>GEE Analysis Polygons</i>, where you can rename or edit it.",
    "From a polygon layer or shapefile":
        "Import polygons from any layer or file (SHP, GPKG, GeoJSON, KML…), all or only the selected ones. "
        "<i>Polygon name</i> is a QGIS expression for the labels.<br><br><b>Click a field on the map</b> "
        "(Field Inspector): click a polygon of the chosen layer; with <i>Inspector mode</i> its time series "
        "is computed at once. Already analysed fields come from the cache.",
    "Field layer from a GEE table asset":
        "Downloads a FeatureCollection asset (any size, paged) as a GeoJSON layer in your QGIS profile and "
        "styles it by the given attribute, e.g. <i>crop</i>. Then use <i>Click a field on the map</i>.",
    "Analysis polygons":
        "All polygons that are sent to Earth Engine. Click one to show its charts and select it on the map; "
        "double click opens the Charts tab.",
    "Time window":
        "A calendar year or any date range. Sentinel-2 surface reflectance starts in March 2017, "
        "Landsat 9 at the end of 2021.",
    "Time series":
        "Mean value per polygon and acquisition date: Sentinel-1 VV/VH backscatter in dB (both pass "
        "directions separately), and a Sentinel-2 and a Landsat 8/9 optical index.<br><br>"
        "<b>Max. scene cloud cover</b> skips whole scenes. <b>Min. valid pixels</b> drops dates where too "
        "little of the polygon was cloud-free — changing it re-filters the charts without a new request.",
    "Sentinel-2 monthly composites (map layers)":
        "Quick-look layers: calendar-month median of Sentinel-2 with the SCL cloud mask. Only the first "
        "month is switched on; tick the others in the layer panel to render them. For other periods, "
        "cloud masks or files use <b>Export imagery</b>.",
    # -- Classification ------------------------------------------------------
    "Engine":
        "<b>Earth Engine</b>: features, training and the map are computed in Google Earth Engine; the map "
        "is shown as a tile layer and can be downloaded.<br><br><b>Local raster stack</b>: training and "
        "classification run on your computer with scikit-learn, on a multi-band GeoTIFF whose band names "
        "are the features. Download a feature stack once (needs Earth Engine) or use any stack.",
    "Training data":
        "Polygons with a known class. The class field may hold numeric codes (kept as map values if 0–254) "
        "or names (coded 0…n−1). <b>Load classes</b> lists them; edit names and double click colours.<br><br>"
        "Keep an independent test set: by an attribute (e.g. spatial blocks) or at random by polygon.",
    "Classification area":
        "Where the map is produced. The optional ESA WorldCover mask keeps only the given land-cover "
        "classes (40 cropland, 10 tree cover, 30 grassland…), so desert, settlements and water don't get "
        "a crop class.",
    "Features (time-series composites)":
        "How the satellite time series is turned into features: for every period of the season, the "
        "chosen variables are aggregated (e.g. median NDVI of May). Band names look like "
        "<i>NDVI_05_median</i> and don't depend on the year, so a model can be applied to other years.",
    "Classifier":
        "Random Forest is the robust default. <b>Variables per split</b> = auto uses √(number of features). "
        "<b>Seed</b> makes runs reproducible — a saved model retrains to the identical classifier.",
    "Performance":
        "How much of your computer and of Earth Engine a run may use at once. Both values are remembered.",
    "Model":
        "A model file (JSON) stores the feature recipe, classifier settings, classes and colours and the "
        "training samples. <b>Apply model</b> rebuilds the same features for another year (or uses a local "
        "stack) and retrains from the stored samples.<br><br><b>Pool samples</b> adds the samples of other "
        "model files (e.g. other years) to the next training run.",
    "Field-level results":
        "Aggregates the classified map to polygons: majority class, its share of the field's pixels "
        "(confidence, <i>cls_share</i>), the runner-up class and the pixel count. Low shares point to mixed "
        "or uncertain fields.",
    "Save the map":
        "The Earth Engine map layer is a tile service and can't be saved with QGIS' <i>Save As</i>. "
        "<b>Download GeoTIFF</b> computes it tile by tile (areas up to a few hundred km² at 10 m); "
        "<b>Drive</b> / <b>asset</b> start an Earth Engine batch task for large areas.",
    "Yearly classified maps":
        "One classified GeoTIFF per year (class codes, 255 = NoData). Class names and colours are read from "
        "the files. All maps are compared on the grid of the earliest year.",
    "Area (optional)":
        "Restricts areas and transitions to these polygons, e.g. a district or an irrigation scheme.",
    # -- Export imagery ------------------------------------------------------
    "Area":
        "The area of the files: a polygon layer / shapefile (clipped to the polygons or their bounding box), "
        "a rectangle drawn on the map, or the current map view.",
    "Composites":
        "The same composite engine as the classification. For each year and period, the clear observations "
        "of the chosen variables are aggregated. The summary below lists every period with its dates.",
    "Output":
        "<b>Files</b>: one stack per year, one file per period, or one time series per variable.<br>"
        "<b>Data type</b>: Float32 keeps physical values; Int16 halves the size (scale factor stored, "
        "QGIS shows physical values).<br>Every file gets band descriptions, band metadata and a JSON "
        "sidecar with the full recipe — and works as a stack for the offline classification.",
}

CORES = ("CPU cores for the <b>offline</b> engine: Random Forest training and prediction run in parallel "
         "(on 24 cores about 12× faster training and 6× faster prediction than on one). Default: all but one, "
         "so QGIS stays responsive. Gradient boosting and single trees use one core. Earth Engine runs on "
         "Google's servers — your cores don't matter there.")
REQUESTS = ("How many Earth Engine requests run at the same time: tile downloads (classified map, feature "
            "stacks, Export imagery) and the folds of an online cross-validation. 4–8 is usually fastest; "
            "Earth Engine limits concurrent requests per user, more can lead to retries or 'too many "
            "requests' errors.")
SPECTRA_CLOUD = ("<b>Cloud mask</b>: Sentinel-2 SCL (cloud, shadow, cirrus, saturated) / Landsat QA_PIXEL "
                 "(cloud, dilated cloud, shadow, cirrus). With the mask, a date only counts if the point is "
                 "clear — or, for polygons, at least the given share of the polygon. <b>Scenes ≤ x % cloud</b> "
                 "skips whole scenes beforehand. Without the mask the closest scene is used as it is.")
PERIOD = ("Length of the composites. Dekads = 1st–10th, 11th–20th, 21st–end of month. With "
          "<i>whole season</i> each variable gets one value per year.")
REDUCERS = ("How the clear observations of a period are combined. Median is robust to remaining clouds; "
            "percentiles and the standard deviation add information about the spread.")
GAP_FILL = ("Periods without any clear observation are filled with the composite of the period ± one period "
            "length. Without filling they stay NoData — such samples are dropped and such pixels are not "
            "classified.")
CLOUD_MASK = ("<b>Cloud Score+</b> (recommended): a pixel is used if its cs_cdf score ≥ the threshold — 0.6 is "
              "a good default, higher is stricter.<br><b>SCL</b>: the Sentinel-2 scene classification; "
              "vegetation, bare soil, water and unclassified pixels are kept.")
SCENE_CLOUD = "Whole scenes with more cloud than this (CLOUDY_PIXEL_PERCENTAGE) are skipped before masking."
S1_ORBIT = ("Ascending and descending passes look from different sides; using both gives more observations, "
            "a single pass gives more consistent values.")
SCALE = "Pixel size used for sampling (classification) or for the files (export)."
SPLIT = ("The test set must be independent of training. <b>By attribute</b>: e.g. a column <i>split</i> with "
         "the value <i>test</i> (ideally spatial blocks). <b>Random by polygon</b>: stable per polygon id.")
SAMPLING = ("<b>Polygon mean</b>: one sample per field — fast, robust, as in the original script. "
            "<b>Pixels</b>: random pixels inside the fields (max. per class) — more samples, but pixels of one "
            "field are similar.")
CV = ("Spatial k-fold cross-validation: the samples are grouped into square blocks of this size by their "
      "polygon centroid and the blocks are split into k folds. Choose blocks larger than typical field "
      "clusters — the result is accuracy as mean ± spread over the folds.")
EXP_LAYOUT = ("<b>Stack</b>: all periods and variables of a year in one file (input for the classification). "
              "<b>Per period</b>: e.g. one file per month. <b>Per variable</b>: a time series, e.g. all NDVI "
              "periods in one file.")
EXP_DTYPE = ("<b>Float32</b>: reflectance 0–1, indices, dB. <b>Int16</b>: optical × 10000, SAR dB × 100, NoData "
             "−32768; the GDAL scale factor is set, so QGIS and the offline classification read physical "
             "values.")
CHECK = ("Training polygons whose features lie far from their own class (robust mean |z| above the threshold) "
         "or closer to another class's centroid. Typical causes: wrong label, mixed or late-sown fields, "
         "clouds. Check them on the map before excluding.")
PROFILES = ("Mean ± 1 standard deviation of the model's training samples per class and period. Where the "
            "bands of two classes overlap over the whole season, the classifier cannot separate them well.")
