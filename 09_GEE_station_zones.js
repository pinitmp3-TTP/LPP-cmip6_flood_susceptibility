// ============================================================
// REVISION 2b - nearest-station (Voronoi) zones for spatial lambda
//   Builds a station_id raster (each pixel = nearest rain gauge) clipped to
//   the LPP watershed, so lambda can vary by sub-region (answers Reviewer Xia #1).
//   Output: MyDrive/GEE_LPP_CMIP6/lambda_zones_station_id.tif (same 30 m grid)
//
//   >>> EDIT the station coordinates below (lat/lon) from the pre-selection
//       paper's station table (Table A2) or TMD/RID metadata. <<<
// ============================================================
var aoi = ee.FeatureCollection('projects/lamphra-pleong-project/assets/Watershed_v2');
var region = aoi.geometry();
var PROJ = 'EPSG:32648', SCALE = 30, FOLDER = 'GEE_LPP_CMIP6';

// Real station coordinates (decimal degrees) from the pre-selection station table.
var stations = ee.FeatureCollection([
  ee.Feature(ee.Geometry.Point([101.683889, 14.492778]), {station_id: 250751}),
  ee.Feature(ee.Geometry.Point([101.725000, 14.355833]), {station_id: 250771}),
  ee.Feature(ee.Geometry.Point([101.651111, 14.476389]), {station_id: 250781}),
  ee.Feature(ee.Geometry.Point([101.566111, 14.861944]), {station_id: 250541}),
  ee.Feature(ee.Geometry.Point([101.478056, 14.592778]), {station_id: 250651}),
  ee.Feature(ee.Geometry.Point([102.001111, 14.942222]), {station_id: 250791})
]);

// nearest-station per pixel: pick the station with the smallest distance
var coll = ee.ImageCollection(stations.map(function (f) {
  var id = ee.Number(f.get('station_id'));
  var dist = ee.FeatureCollection([f]).distance(300000);          // metres
  return dist.multiply(-1).rename('negdist')
             .addBands(ee.Image.constant(id).toInt32().rename('station_id'));
}));
var nearest = coll.qualityMosaic('negdist').select('station_id').clip(region);

Map.centerObject(aoi, 11);
Map.addLayer(nearest.randomVisualizer(), {}, 'Nearest-station zones');
Map.addLayer(stations, {color: 'black'}, 'Stations');

Export.image.toDrive({
  image: nearest, description: 'lambda_zones_station_id', folder: FOLDER,
  region: region, scale: SCALE, crs: PROJ, maxPixels: 1e13
});
// AFTER RUN: Tasks -> RUN. File -> MyDrive/GEE_LPP_CMIP6/lambda_zones_station_id.tif
// Then run 10_stageII_d_spatial_lambda_recalibration.py
