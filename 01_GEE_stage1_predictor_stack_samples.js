// ============================================================
// Paper #5 - CMIP6 flood susceptibility, Lam Phra Phloeng
// STAGE 1 (Google Earth Engine)  -  PATH A, two-stage design
//
//   Stage I  (ML)      : spatial susceptibility S(x) from STATIC factors + LULC
//   Stage II (climate) : basin-uniform R10mm/R95p -> magnitude/frequency scaling
//                        (handled OUTSIDE GEE as a small scalar table, NOT a raster)
//
// Builds ONLY Stage-I inputs: static predictor stack (+ LULC) -> training samples.
// Rainfall indices are deliberately NOT pixel predictors here.
// BOTH exports go to the Drive folder set in DRIVE_FOLDER below.
// ============================================================

// ------------------------------------------------------------
// 0.  USER CONFIG  -  >>> EDIT THESE <<<
// ------------------------------------------------------------
var aoi          = ee.FeatureCollection('projects/lamphra-pleong-project/assets/Watershed_v2'); // >>> watershed boundary
var floodFreq    = ee.Image('projects/lpp-project-508605/assets/LPP_frequency_3events');         // >>> SAR flood-frequency (0-3)
// var geology   = ee.Image('users/EDIT/lpp_geology');                                           // >>> optional geology raster

var DRIVE_FOLDER = 'GEE_LPP_CMIP6';   // <<< all exports land in this Drive folder

var PROJ   = 'EPSG:32648';   // UTM zone 48N (Nakhon Ratchasima)
var SCALE  = 30;             // common analysis grid (m)
var SEED   = 42;
var N_PER_CLASS = 2000;      // samples per class (flood / non-flood); tune after first run
var region = aoi.geometry();

// ------------------------------------------------------------
// 1.  TERRAIN (Copernicus GLO-30, reprojected to metric grid)
// ------------------------------------------------------------
var dem = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM').mosaic()
            .setDefaultProjection('EPSG:4326', null, 30)
            .reproject({crs: PROJ, scale: SCALE})
            .rename('elevation').clip(region);

var slope     = ee.Terrain.slope(dem).rename('slope');       // degrees
var aspect    = ee.Terrain.aspect(dem).rename('aspect');     // degrees (circular -> sin/cos in Python)
var curvature = dem.convolve(ee.Kernel.laplacian8()).rename('curvature');

// ------------------------------------------------------------
// 2.  HYDROLOGY (MERIT Hydro) + derived indices
// ------------------------------------------------------------
var merit = ee.Image('MERIT/Hydro/v1_0_1');
var hand  = merit.select('hnd').rename('hand');              // Height Above Nearest Drainage (m)
var upa   = merit.select('upa').rename('upa');               // upstream drainage area (km^2)

var slopeRad = slope.multiply(Math.PI / 180);
var tanSlope = slopeRad.tan().max(0.001);                    // guard against tan(0)
var sca      = upa.multiply(1e6).divide(SCALE);              // specific catchment area (m)

var twi = sca.divide(tanSlope).log().rename('twi');                        // ln(a / tan b)
var spi = sca.multiply(tanSlope).add(1).log().rename('spi');              // ln(a * tan b + 1)

var channel   = upa.gt(1);                                                 // channel = upstream area > 1 km^2 (tune)
var distRiver = channel.fastDistanceTransform(256).sqrt()
                   .multiply(SCALE).rename('dist_river');                  // approx. distance (m)
var drainDens = channel.reduceNeighborhood({
                   reducer: ee.Reducer.mean(),
                   kernel:  ee.Kernel.circle(1000, 'meters')
                 }).rename('drain_density');

// ------------------------------------------------------------
// 3.  SURFACE predictors (soil + LULC)
//     LULC stays in the ML: it is spatial AND its future change is the source
//     of spatial variation in the change map (factorial attribution, Stage 4).
// ------------------------------------------------------------
var soil = ee.Image('OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02')
             .select('b0').rename('soil');                   // categorical -> one-hot in Python

var lulc = ee.ImageCollection('ESA/WorldCover/v200').first()
             .rename('lulc');                                // present LULC (categorical). Held constant in primary runs;
                                                             // swapped to CA-Markov 2045 for the LULC-attribution run.

// ------------------------------------------------------------
// 4.  PREDICTOR STACK (11 static bands - NO rainfall)
// ------------------------------------------------------------
var predictors = ee.Image.cat([
  dem, slope, aspect, curvature,
  hand, twi, spi, distRiver, drainDens,
  soil, lulc
  // , geology.rename('geology')                             // uncomment if geology asset provided
]).clip(region);

// ------------------------------------------------------------
// 5.  ANALYSIS DOMAIN  (matches SAR terrain mask: lowland only)
// ------------------------------------------------------------
var jrc       = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence');
var permWater = jrc.gt(50).unmask(0);
var domain    = hand.lt(15).and(slope.lt(5)).and(permWater.not());

// ------------------------------------------------------------
// 6.  LABELS  (1 = flood ; 0 = non-flood, buffered from omitted floods)
// ------------------------------------------------------------
var floodMask   = floodFreq.gte(1).and(domain);                            // presence
var distFlood   = floodFreq.gte(1).fastDistanceTransform(256).sqrt().multiply(SCALE);
var absenceMask = domain.and(floodMask.not()).and(distFlood.gt(150));      // >=150 m from any flood pixel

var label = ee.Image(0).byte().rename('label')
              .where(floodMask, 1)
              .updateMask(floodMask.or(absenceMask));                      // keep only class 0/1 pixels

// ------------------------------------------------------------
// 7.  STRATIFIED SAMPLE  (balanced 1:1) + coords + frequency
// ------------------------------------------------------------
var sampleStack = predictors
                    .addBands(ee.Image.pixelLonLat())        // 'longitude','latitude' for spatial-block CV in Python
                    .addBands(floodFreq.rename('freq'))      // 1/2/3 -> optional graded target / sample weight
                    .addBands(label);

var samples = sampleStack.stratifiedSample({
  numPoints:  N_PER_CLASS,
  classBand:  'label',
  region:     region,
  scale:      SCALE,
  projection: PROJ,
  seed:       SEED,
  geometries: true,
  tileScale:  4
});

print('Sample size (should be ~2x N_PER_CLASS):', samples.size());
print('Class balance:', samples.aggregate_histogram('label'));
print('First sample:', samples.first());

// ------------------------------------------------------------
// 8.  EXPORTS  ->  Drive folder GEE_LPP_CMIP6
// ------------------------------------------------------------
var cols = ['elevation','slope','aspect','curvature','hand','twi','spi',
            'dist_river','drain_density','soil','lulc',
            'longitude','latitude','freq','label'];

Export.table.toDrive({
  collection:  samples,
  description: 'LPP_paper5_samples_static',
  folder:      DRIVE_FOLDER,                     // <<< lands in GEE_LPP_CMIP6
  fileFormat:  'CSV',
  selectors:   cols
});

Export.image.toDrive({
  image:       predictors.updateMask(domain).toFloat(),
  description: 'LPP_paper5_predictor_stack_static',
  folder:      DRIVE_FOLDER,                     // <<< lands in GEE_LPP_CMIP6
  region:      region,
  scale:       SCALE,
  crs:         PROJ,
  maxPixels:   1e13
});

// ------------------------------------------------------------
// 9.  QUICK VISUAL QA
// ------------------------------------------------------------
Map.centerObject(aoi, 11);
Map.addLayer(dem,        {min:180, max:500, palette:['225ea8','41b6c4','ffffcc','fd8d3c','bd0026']}, 'Elevation');
Map.addLayer(domain.selfMask(),      {palette:['1d9e75']}, 'Analysis domain', false);
Map.addLayer(floodMask.selfMask(),   {palette:['185fa5']}, 'Flood (label 1)');
Map.addLayer(absenceMask.selfMask(), {palette:['993c1d']}, 'Non-flood zone (label 0)', false);
Map.addLayer(samples, {color:'black'}, 'Samples', false);

// ------------------------------------------------------------
// AFTER RUNNING: open the "Tasks" tab (top-right) and click RUN on both tasks
//   - LPP_paper5_samples_static           -> CSV for Stage 2
//   - LPP_paper5_predictor_stack_static   -> stack for Stage 3
// Both appear in Drive under: MyDrive/GEE_LPP_CMIP6/
// ============================================================
