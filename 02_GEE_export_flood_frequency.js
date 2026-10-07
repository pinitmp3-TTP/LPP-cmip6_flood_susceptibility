// ============================================================
// Paper #5 - export flood-frequency inventory as GeoTIFF
// One-time export for Stage 3b validation.
// Same grid as the predictor stack: EPSG:32648, 30 m, clipped to watershed.
// ============================================================

var aoi       = ee.FeatureCollection('projects/lamphra-pleong-project/assets/Watershed_v2'); // watershed
var floodFreq = ee.Image('projects/lpp-project-508605/assets/LPP_frequency_3events');         // SAR freq map (0-3)

// quick look
Map.centerObject(aoi, 11);
Map.addLayer(floodFreq.selfMask(), {min: 1, max: 3, palette: ['ffff00', 'ff8000', 'ff0000']},
             'Flood frequency (1-3)');

// export -> MyDrive/GEE_LPP_CMIP6/LPP_flood_frequency.tif
Export.image.toDrive({
  image:       floodFreq.rename('freq').clip(aoi.geometry()).toFloat(),
  description: 'LPP_flood_frequency',
  folder:      'GEE_LPP_CMIP6',
  region:      aoi.geometry(),
  scale:       30,
  crs:         'EPSG:32648',
  maxPixels:   1e13
});

// AFTER RUN: open the "Tasks" tab (top-right) and click RUN on "LPP_flood_frequency".
// Wait until it turns green, then the file appears in MyDrive/GEE_LPP_CMIP6/.
