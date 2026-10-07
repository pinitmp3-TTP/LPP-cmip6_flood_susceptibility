// ============================================================
// REVISION - reservoir zones + exposure layers (answers Mitchell #2, Mohanty #3)
//   Exports, on the 30 m analysis grid:
//     reservoir_zone.tif  : 1=upstream of dam, 2=downstream of dam, 3=reservoir pool
//                           (drawdown zone, to separate from genuine flooding)
//     population.tif       : WorldPop population count (exposure)
//   Cropland exposure comes from the LULC band already in the predictor stack.
// ============================================================
var aoi = ee.FeatureCollection('projects/lamphra-pleong-project/assets/Watershed_v2');
var region = aoi.geometry();
var PROJ='EPSG:32648', SCALE=30, FOLDER='GEE_LPP_CMIP6';
var DAM = ee.Geometry.Point([101.781, 14.505]);        // Lam Phra Phloeng dam

var merit = ee.Image('MERIT/Hydro/v1_0_1');
var hand  = merit.select('hnd');
var elev  = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM').mosaic()
  .setDefaultProjection('EPSG:4326',null,30).reproject({crs:PROJ,scale:SCALE}).clip(region);

// reservoir pool = persistent surface water (JRC occurrence > 50%)
var pool = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').gt(50).unmask(0);

// upstream vs downstream: elevation of the dam, pixels higher = upstream
var damElev = ee.Number(elev.reduceRegion({reducer:ee.Reducer.first(),geometry:DAM,scale:SCALE})
                 .values().get(0));
var upstream = elev.gte(ee.Image.constant(damElev));

// ROUND-2 (Mitchell): seasonal DRAWDOWN ring = water present 5-50% of the time and
// within 3 km of the permanent pool. These margins stay in the domain, so we flag
// them as zone 4 and report how much of the flood inventory falls there.
var occ = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').unmask(0);
var nearPool = pool.selfMask().focal_max({radius: 3000, kernelType: 'circle', units: 'meters'})
                 .unmask(0);
var drawdown = occ.gte(5).and(occ.lte(50)).and(nearPool);

var zone = ee.Image(2).where(upstream, 1).where(drawdown, 4).where(pool, 3).toInt16()
             .updateMask(hand.lt(15)).rename('reservoir_zone').clip(region);

// exposure: WorldPop population count (most recent)
var pop = ee.ImageCollection('WorldPop/GP/100m/pop').filterBounds(region)
            .filter(ee.Filter.calendarRange(2020,2020,'year')).mosaic()
            .reproject({crs:PROJ,scale:SCALE}).rename('population').clip(region);

Map.centerObject(aoi,11);
Map.addLayer(zone.randomVisualizer(),{},'Reservoir zones (1 up,2 down,3 pool,4 drawdown)');
Map.addLayer(pop,{min:0,max:50,palette:['white','orange','red']},'Population',false);

Export.image.toDrive({image:zone,description:'reservoir_zone',folder:FOLDER,
  region:region,scale:SCALE,crs:PROJ,maxPixels:1e13});
Export.image.toDrive({image:pop.unmask(0),description:'population',folder:FOLDER,
  region:region,scale:SCALE,crs:PROJ,maxPixels:1e13});
// ROUND-2: MERIT Hydro D8 flow direction on its NATIVE 3-arcsec grid (never resample
// flow directions). Used by 22_dam_catchment_updown.py to delineate the dam catchment,
// replacing the elevation rule above, which put reservoir margins below the crest
// elevation into "downstream".
Export.image.toDrive({image:merit.select('dir').toUint8(),description:'merit_flowdir',folder:FOLDER,
  region:region.buffer(3000),crs:'EPSG:4326',
  crsTransform:[0.000833333333333333,0,-180,0,-0.000833333333333333,90],maxPixels:1e13});
// RUN -> Tasks -> RUN all three. Files -> GEE_LPP_CMIP6/.  Then run 23_reservoir_exposure_summary.py
