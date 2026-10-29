# GEE Time Series Analyzer & Classifier (QGIS plugin)

Detailed reference of all options. For an overview with screenshots, installation and the privacy notes see the project README in the repository root.

A QGIS port of two Google Earth Engine Code Editor apps, a **time series analyzer** and a **field inspector**. Instead of the Code Editor map and `ui.Chart`s, it drives the real QGIS map canvas and hosts interactive matplotlib charts in a dock widget. Any polygon anywhere works.

## What it does

The dock has three sections, switched in its header:

-   **Time series**: the *Polygons*, *Analysis*, *Charts* and *Compare* tabs.
-   **Classification**: the *Setup*, *Results*, *Class profiles*, *Training check* and *Crop rotation* tabs.
-   **Export imagery**: composites and time-series stacks as GeoTIFF.

-   **Polygons tab**: where the polygons come from.
    -   **Draw** any number of polygons on the canvas (left click = vertex, right/double click or Enter = finish, Backspace = undo vertex, Esc = discard). Each one gets its own colour (the first three are the original app's red/teal/blue).
    -   **Shapefile / any polygon layer**: "Open file…" (SHP, GPKG, GeoJSON, KML, …) imports every polygon directly. Or pick a project layer and import all or only the selected features. Polygon names come from a QGIS expression (e.g. `'Field ' || "fid" || ' (' || "crop" || ')'`).
    -   **Click a field on the map** (Field Inspector mode): click a polygon of the chosen layer. With *Inspector mode* on, the analysis starts right away. With *Keep previously clicked fields* on (the default), every clicked field stays in the list. Only the new field is queried from GEE; the others come from the cache, so switching back is instant. Clicking an already analysed field just jumps to it, and clicks made while an extraction is running are queued.
    -   **GEE table asset**: downloads a FeatureCollection asset (paged, so the 5000-feature limit doesn't apply) as a real layer styled by crop type with the Field Inspector's palette. The three field assets of the original app are preset, and picking a preset also sets its year.
    -   All analysis polygons live in an editable memory layer, **"GEE Analysis Polygons"**, so you can rename them in the attribute table, move vertices, or save the layer.
-   **Analysis tab**
    -   Year (2015–2026) or a custom date range.
    -   Time series: **Sentinel-1 VV + VH, descending and ascending** (dB, GRD border noise < −30 dB masked). **Sentinel-2 SR** (SCL cloud mask) and **Landsat 8/9 C2 L2** (QA_PIXEL cloud/shadow mask), each with a selectable optical index: NDVI, EVI, SAVI, NDWI or NDMI.
    -   Scene cloud pre-filter, plus a minimum share of valid pixels *inside each polygon*. Change the share afterwards and the charts update instantly, with no new GEE request.
    -   **Sentinel-2 monthly median composites** as map layers (default May, July and September, as in the original). Choose true colour, false-colour NIR or NDVI, and the extent: the polygons plus a buffer, or the current map view. They go into a "GEE Sentinel-2 composites" layer group and are only rebuilt when the year, months or extent change. Only the first month is switched on; the others are in the group but are not rendered until you tick them.
-   **Export imagery** section: composites and time-series stacks as GeoTIFF, with the same composite engine as the Classify tab and every setting visible.
    -   **Area**: a polygon layer or shapefile (*Open file…*, optionally selected features only; clipped to the polygons or their bounding box), a **rectangle drawn on the map**, or the current map view.
    -   **Composites**:
        -   years (one or a range, files per year) and season (start/end month; Oct–Jun starts in the previous year)
        -   aggregation period: dekads, semi-monthly, monthly, every 2 or 3 months, or the whole season
        -   aggregation: median, mean, min, max, p10/p25/p75/p90, standard deviation, in any combination
        -   gap filling from the neighbouring periods, or none
        -   Sentinel-2 indices (NDVI, EVI, SAVI, NDWI, NDMI, NDRE) and bands B2–B12, with the cloud mask (Cloud Score+ with threshold, or SCL) and the maximum scene cloud cover
        -   Sentinel-1 VV, VH, VH−VV and the orbit
        -   resolution
    -   **Output**:
        -   content: data, or RGB quicklooks in true or false colour (8-bit, one file per period)
        -   files: one multi-band stack per year, one file per period, or one file per variable (a time series)
        -   data type: Float32 with physical values (NoData −9999), or Int16 scaled (optical × 10000, SAR dB × 100) with the GDAL scale factor set, so QGIS and the offline classification read physical values
        -   optional ESA WorldCover band, CRS (default: UTM of the area) and file name prefix
    -   **Summary** (updated live): every period with its dates, aggregation, cloud mask and scene filter, S1 settings, number of files and bands, an example file name, the area and the estimated size. It warns when the area is too large for the direct download.
    -   **Files**:
        -   Band descriptions are the feature names (e.g. `NDVI_05a_median`).
        -   Each band carries metadata: variable, sensor, period, first and last day, reducer.
        -   The dataset carries year, recipe and layout.
        -   Each file gets a JSON sidecar with the full recipe and band table.
        -   Every data file is a valid stack for the offline classification.
    -   *Download GeoTIFFs…* computes the files tile by tile; this suits areas up to a few hundred km² at 10 m. *Export to Drive* starts one batch task per file.
-   **Charts tab** (index panels such as NDVI show a dashed 0 line; ◀ ▶ buttons, a click in the polygon list, or selecting the polygon on the map switches the view instantly): "Combined — all polygons" (one line per polygon, with Landsat dashed as in the original) and one six-panel chart per polygon. Pan, zoom and save-PNG are on every chart. **CSV export** and **export all charts as PNG**.
-   **Compare tab**: tick any number of analysed polygons (*All*, *None*, or *Map selection* to take the polygons selected on the map; *Clear selection* removes the highlight and the map selection but keeps the ticks) and chart them on shared axes, one series or all series stacked. You get uniform grey lines (or polygon colours) and an optional mean ± 1σ band. The list ranks polygons by **deviation** (mean |z-score| against the ticked polygons on the same dates), so outliers come first. **Click a line** (or a list row) to highlight it in red; its polygon is selected, flashed and panned to on the map. Selecting a polygon on the map highlights its line in return. Hovering over a line shows its name.
-   **Classification** section: supervised crop classification in Earth Engine. It is a port and generalisation of a Code Editor crop-classification script.
    -   **Training data**: any polygon layer in QGIS (all or selected features) or a GEE table asset. The class field can hold numeric codes, which stay as map values if they are 0–254 (e.g. `class` 0–9), or names, which are coded 0…n−1. An optional name field (e.g. `crop`) supplies readable names. *Load classes* lists the classes with their polygon counts. You can rename them and double click a colour to change it.
    -   **Train/test split**: by an attribute (e.g. `split` = `test`, spatial blocks), random by polygon, or none.
    -   **Samples**: one per polygon (polygon mean, as in the script), or random pixels inside the polygons (max. N per class).
    -   **Classification area**: another polygon layer (dissolved), a GEE asset (e.g. a district boundary) or the current map view. An optional ESA WorldCover mask (default 40 cropland + 10 tree cover) keeps desert, settlements and water out of the crop classes.
    -   **Features**:
        -   Year and season (start/end month; a season such as Oct–Jun starts in the previous year).
        -   Aggregation period: monthly, semi-monthly, every 2 or 3 months.
        -   Aggregation: any combination of median, mean, min, max, p10/p25/p75/p90 and standard deviation.
        -   Gap filling: neighbouring periods, or none.
        -   Sentinel-2: NDVI, EVI, SAVI, NDWI, NDMI, NDRE and/or bands B2–B12, with a Cloud Score+ (threshold) or SCL cloud mask.
        -   Sentinel-1: VV, VH and/or VH−VV, from either or both passes.
        -   The tab shows how many features the settings produce.
    -   **Classifier**: Random Forest (trees, variables per split, min. leaf population, bag fraction, max. nodes, seed), Gradient Tree Boost or CART.
    -   **Results** (inner *Results* tab):
        -   The map is added to the "GEE classification" group in the class colours.
        -   Overall accuracy and kappa for the test samples. With polygon-mean samples, also for the map pixels inside the test polygons, as in the script's section 8d.
        -   Out-of-bag error, confusion matrices as heatmaps, producer's/user's accuracy and F1 per class, variable importance.
        -   *Save validation tables* writes metrics, confusion matrices, per-sample predictions and importance as CSV.
        -   *Export HTML report…* writes one self-contained HTML file and opens it in the browser. It contains:
            -   summary cards: OA with 95 % confidence interval, kappa with the Landis & Koch scale, balanced accuracy, macro F1, quantity and allocation disagreement (Pontius & Millones 2011), OOB error
            -   a sortable per-class table: colour, training samples with a sample-size rating, producer's/user's accuracy, F1, omission and commission errors
            -   colour-coded confusion matrices as HTML tables (samples and map pixels), with totals and PA/UA
            -   a list of the most frequent confusions
            -   charts and the full run settings (periods, cloud mask, classifier, split), plus references
        -   *Export trees (HTML)…* writes an interactive viewer of every tree of a Random Forest or CART model, from Earth Engine or offline. The trees are rebuilt from the samples stored in the model with the same seed, so they are exactly the classification's trees.
            -   **Forest overview**: number of trees, mean depth and leaves, and the usage of each variable across all splits (count, share, root splits, mean depth). Click a variable to highlight its splits.
            -   **Tree view**: a tree list (filter by root variable, sort by depth or size) and a zoomable, pannable view (mouse wheel, drag).
            -   Nodes open and close with a click, *open to depth* sets how far the tree is expanded, and *Expand all* opens every node.
            -   Each node shows its split (left = ≤ threshold), the number of samples and a bar with the class mix; leaves are coloured by class, and hovering shows the full class probabilities.
            -   300 Earth Engine trees from ~1100 field samples give about 8 MB.
            -   Gradient boosting is not supported, because its trees are regression trees per class.
        -   The plots in the plugin grow with the number of classes: the matrix height, horizontal per-class bars with the class colours, and sideways scrolling if needed.
        -   The map layer itself is an Earth Engine tile service, so QGIS' *Export → Save As* can't write it (error "0x0 dataset"). *Download GeoTIFF…* computes the map tile by tile and saves a local GeoTIFF instead: class codes, 255 = NoData, colour table, class names (in the `.aux.xml` next to the file), overviews. The file is added as a normal raster layer with a paletted legend. This suits areas up to a few hundred km² at 10 m; an 8 × 9 km area takes ~40 s.
        -   *Export the map* starts an Earth Engine batch task (for large areas): a Cloud-Optimised GeoTIFF to Drive (background = 255 = NoData, UTM of the area by default) or an image asset (masked background).
    -   **Offline mode** (*Engine: Local raster stack*): training, accuracy assessment and classification run locally with scikit-learn on GeoTIFF raster stacks, without Earth Engine.
        -   A *stack* is a multi-band GeoTIFF whose band descriptions are the feature names.
        -   *Download feature stack…* uses Earth Engine once. It writes the features set on the tab (year, season, periods, reducers, indices/bands) as float32 with NoData −9999, plus the ESA WorldCover classes as band `worldcover` (used for the mask). Year and feature recipe are stored in the file.
        -   Stack area: the classification area (clipped to it), the bounding box of the training polygons, or both. Use "both" if one stack serves for training and prediction.
        -   Training stack and stack to classify can be the same file or two files, e.g. a small stack around the fields and a large one for the district.
        -   Any other stack works too, e.g. one from **Export imagery** (Float32 or scaled Int16).
        -   The training polygons must be a QGIS layer; download an asset on the Polygons tab first. They are rasterised onto the stack grid. Samples are polygon means or pixels, and the split, accuracy and CSV outputs are the same as online.
        -   The classifier settings map to scikit-learn: Random Forest (trees, max. features, min. samples per leaf, bootstrap fraction, max. leaf nodes, seed), Gradient Boosting or a decision tree.
        -   The map is written block by block to a paletted GeoTIFF (255 = NoData) and added to the project.
        -   Models have the same file format in both modes. An online model can classify a local stack with the same band names, and an offline model trained on a downloaded feature stack can be applied online to other years. Because both modes retrain from the stored samples, the results differ slightly: GEE's smile RF and scikit-learn's RF are different implementations.
        -   scikit-learn is required for the offline mode. If it is missing, run `python -m pip install scikit-learn` in the OSGeo4W Shell.
    -   **Spatial k-fold cross-validation** (option under *Classifier*): the train + test samples are grouped into square blocks (e.g. 5 km) by their polygon centroid, and the blocks are split into k folds. Neighbouring fields and the pixels of one field never end up in training and test at once. The result is OA and kappa as mean ± sd over the folds, per-class F1 mean ± sd and a per-fold table, shown in Results and in the HTML report. Works online and offline.
    -   **Multi-year training** (*Model → Pool samples from models…*): the samples stored in other model files (e.g. a 2025 model) are added to the next training run. They need the same feature bands; classes are matched by value or name. The pooled sources are recorded in the model.
    -   **Field-level results** (*Results*): aggregates the map, from Earth Engine or a local GeoTIFF, to any polygon layer. Each field gets the majority class (`cls_code`, `cls_name`), its share of the field's pixels as a confidence (`cls_share`), the runner-up class (`cls2_*`) and `n_pixels`. The output is a class-styled layer; filter on `cls_share` to find mixed or uncertain fields.
    -   **Class profiles** (inner tab): mean ± 1σ of the model's training samples per class over the periods (real dates if the model has a recipe), per variable/reducer, with classes switchable. Overlapping bands explain the confusions. Also in the HTML report.
    -   **Training check** (inner tab): flags training polygons whose features are far from their own class (robust mean |z| above a threshold) or closer to another class's centroid, i.e. likely mislabelled or mixed fields. Click a row to zoom to the field; *Select flagged on map*; *Exclude flagged & retrain* (exclusions persist until cleared and also work for asset training data via `system:index`). The random train/test split is stable per polygon, so exclusions don't reshuffle the other fields.
    -   **Model export/import**: *Save model…* writes a JSON file with the feature recipe, the classifier settings, the classes and colours, and the training sample table. *Load model…* + *Apply model* rebuilds the same features for **any other year** over the chosen classification area. It then retrains the identical classifier from the stored samples (same seed) and classifies. The optional *Validate with the training data above* checks the applied map against reference polygons of that year. Classes are matched by value or by name, so a 2025 layer with only a `crop` column works against a model trained on `class` codes.
-   **Crop rotation** tab (Classification section): crop rotation and change between yearly classified maps.
    -   List one classified GeoTIFF per year with *Add files…* (the year is guessed from the file name) or by choosing layers. Optionally restrict to polygons, e.g. a district.
    -   *Areas*: class areas in ha per year, as a table and stacked bars.
    -   *Transitions*: the matrix for each pair of consecutive years in ha or % of the earlier class, colour-coded, with the share unchanged and the largest changes. *Write change map* writes a GeoTIFF with one value per transition ("cotton → rice"), unchanged pixels grey.
    -   *Flow diagram*: an alluvial chart over all years.
    -   *Save tables* writes the areas and transitions as CSV plus the diagram as PNG.
    -   The maps are compared on the grid of the earliest year; others are resampled (nearest neighbour) if needed.
-   **Processing algorithms** (search "GEE" in the Processing Toolbox; all usable in batch mode and Model Builder):
    -   *Apply classification model (year / stack)*: a model file applied in Earth Engine to a year and area (downloaded as GeoTIFF), or offline to a local raster stack. In batch mode it handles several years at once.
    -   *Aggregate classified map to fields (majority class)*
    -   *Crop transition matrix (two classified maps)*
-   Earlier algorithms: *Extract polygon time series* produces a table of polygon/series/date/value/valid_fraction, ideal for hundreds of fields or Model Builder. *Download GEE table asset as vector layer* is also there.

### How it is computed

The original apps called `ui.Chart.image.series` once per polygon and product. This plugin instead runs one server-side `reduceRegions` per product over **all** polygons at once, then fetches the table with `ee.data.computeFeatures` in pages. Every row also carries the polygon's valid-pixel fraction for that image, so the cloud threshold is applied client-side. Acquisitions from the same day (overlapping S2 tiles, adjacent S1 slices) are merged into one daily mean. A full year of all four products for a few fields takes about 10 s. The extraction runs as a background QGIS task with a Cancel button, so QGIS stays usable.

## Installation

1.  Install and enable the official **Google Earth Engine** plugin (Plugins → Manage and Install Plugins) and sign in once. This plugin uses its bundled `earthengine-api`, its login and its Cloud project.
2.  Copy the `gee_timeseries_analyzer` folder into your QGIS profile's plugins directory, `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\`. Or use *Install from ZIP* with `gee_timeseries_analyzer.zip`.
3.  Enable **GEE Time Series Analyzer & Classifier** and click its toolbar icon.

`pandas` and `matplotlib` are already bundled with QGIS on Windows. If they are missing: `python -m pip install pandas matplotlib` in the OSGeo4W shell.

## Quick start

-   **Free polygons**: Polygons tab → *Draw polygon on map* → draw 1–n polygons → Analysis tab → *Analyze*.
-   **Shapefile**: *Open file…* → *Analyze*.
-   **Field inspector**: choose the asset → *Download asset as layer* → *Click a field on the map* → click fields; the charts update after each click.

## Notes / known limitations

-   Classification: sampling the training polygons is an interactive Earth Engine request. Polygon means of ~1500 fields with 8 monthly NDVI composites take about 2–3 minutes. Very many polygons combined with many features can hit Earth Engine's interactive limits (5 min / memory). Then use fewer features, pixel samples, or a smaller training set. For thousands of polygons, use a GEE table asset as the training source instead of a QGIS layer, which has to be uploaded with every request. Map tiles are computed on the fly, so for large areas the batch export is the way to get the full map.

-   Sentinel-2 SR Harmonized starts in March 2017 and Landsat 9 in late 2021. Earlier years simply show fewer or no optical points.
-   Values are area-weighted polygon means at 10 m (S1, S2) and 30 m (Landsat). For polygons much smaller than a pixel, Landsat may produce no value.
-   Composite layers are added through the Google Earth Engine plugin's `Map.addLayer`. If it is not available, a plain XYZ layer is used instead; its tile URL expires after a few hours.
-   Up to 300 analysis polygons in the dock. Cached results are reused as long as dates, products, index, cloud filter and the polygon's geometry are unchanged. For more, use the Processing algorithm.
-   The memory layer "GEE Analysis Polygons" is not saved with the project unless you make it permanent (right click → Make Permanent).

## Attribution

Copernicus Sentinel-1/-2 data (ESA), Landsat 8/9 Collection 2 (USGS), processed in Google Earth Engine.

## License

GPL-2.0-or-later (see `LICENSE`).

