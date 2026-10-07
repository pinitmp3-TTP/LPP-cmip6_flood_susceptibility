// ============================================================
// Paper #5 - Stage 4a v2: export Dynamic World LULC with MULTI-YEAR composites
// (reduces inter-annual "flicker" of single-year modes).
//
//   Each epoch = modal class over a 3-year window:
//     2016 epoch -> mode(2015-2017), 2020 -> mode(2019-2021), 2024 -> mode(2023-2025)
//   Output (MyDrive/GEE_LPP_CMIP6/LULC/):  DW_lulc_2016.tif, _2020.tif, _2024.tif
//   ESA WorldCover classes, EPSG:32648, 30 m, clipped to watershed.
// ============================================================

var aoi = ee.FeatureCollection('projects/lamphra-pleong-project/assets/Watershed_v2');
var region = aoi.geometry();

var PROJ = 'EPSG:32648';
var SCALE = 30;
var DRIVE_FOLDER = 'GEE_LPP_CMIP6/LULC';     // capital LULC
var EPOCHS = [2016, 2020, 2024];
var HALF = 1;                                 // +/- years around each epoch (3-year window)

var dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1');
var DW_FROM = [0, 1, 2, 3, 4, 5, 6, 7, 8];
var WC_TO   = [80, 10, 30, 90, 40, 20, 50, 60, 70];

function epochWC(center) {
  var start = ee.Date.fromYMD(center - HALF, 1, 1);
  var end   = ee.Date.fromYMD(center + HALF, 12, 31);
  var label = dw.filterBounds(region).filterDate(start, end)
                .select('label').reduce(ee.Reducer.mode());    // modal class over 3 years
  return label.remap(DW_FROM, WC_TO).rename('lulc').toInt16().clip(region);
}

var palette = {min: 10, max: 90,
  palette: ['006400', 'ffbb22', 'ffff4c', 'f096ff', 'fa0000',
            'b4b4b4', 'f0f0f0', '0064c8', '0096a0']};

EPOCHS.forEach(function (y) {
  var img = epochWC(y);
  Map.addLayer(img, palette, 'LULC ' + y, y === 2024);
  Export.image.toDrive({
    image:       img,
    description: 'DW_lulc_' + y,
    folder:      DRIVE_FOLDER,
    region:      region,
    scale:       SCALE,
    crs:         PROJ,
    maxPixels:   1e13
  });
});

Map.centerObject(aoi, 11);
// AFTER RUN: Tasks tab -> RUN all three. Files OVERWRITE the single-year ones
// in MyDrive/GEE_LPP_CMIP6/LULC/. Then re-run 15_stage4b_ca_markov_2045.py.
