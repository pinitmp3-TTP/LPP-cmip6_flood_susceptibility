// ============================================================
// REVISION - export absence samples at buffer 0 m and 500 m (answers Xia #3)
//   The 150 m set already exists (LPP_paper5_samples_static.csv). This adds the
//   two end-members for the buffer-sensitivity test.
//   Only the labelling/sampling differs from Stage 1; predictors are identical.
//   >>> This is the SAME Stage-1 script with BUFFER changed; run it TWICE,
//       once with BUFFER_M = 0 and once with BUFFER_M = 500. <<<
// ============================================================
var BUFFER_M = 0;            // <<< set to 0, run+export; then set to 500, run+export
var SUFFIX   = BUFFER_M === 0 ? 'buffer0' : 'buffer500';

var aoi       = ee.FeatureCollection('projects/lamphra-pleong-project/assets/Watershed_v2');
var floodFreq = ee.Image('projects/lpp-project-508605/assets/LPP_frequency_3events');
var PROJ='EPSG:32648', SCALE=30, SEED=42, N=2000, region=aoi.geometry();

var dem=ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM').mosaic()
  .setDefaultProjection('EPSG:4326',null,30).reproject({crs:PROJ,scale:SCALE}).rename('elevation').clip(region);
var slope=ee.Terrain.slope(dem).rename('slope'), aspect=ee.Terrain.aspect(dem).rename('aspect');
var curvature=dem.convolve(ee.Kernel.laplacian8()).rename('curvature');
var merit=ee.Image('MERIT/Hydro/v1_0_1'), hand=merit.select('hnd').rename('hand'), upa=merit.select('upa').rename('upa');
var tanS=slope.multiply(Math.PI/180).tan().max(0.001), sca=upa.multiply(1e6).divide(SCALE);
var twi=sca.divide(tanS).log().rename('twi'), spi=sca.multiply(tanS).add(1).log().rename('spi');
var channel=upa.gt(1);
var distRiver=channel.fastDistanceTransform(256).sqrt().multiply(SCALE).rename('dist_river');
var drainDens=channel.reduceNeighborhood({reducer:ee.Reducer.mean(),kernel:ee.Kernel.circle(1000,'meters')}).rename('drain_density');
var soil=ee.Image('OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02').select('b0').rename('soil');
var lulc=ee.ImageCollection('ESA/WorldCover/v200').first().rename('lulc');
var predictors=ee.Image.cat([dem,slope,aspect,curvature,hand,twi,spi,distRiver,drainDens,soil,lulc]).clip(region);

var jrc=ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence');
var domain=hand.lt(15).and(slope.lt(5)).and(jrc.gt(50).unmask(0).not());
var floodMask=floodFreq.gte(1).and(domain);
var distFlood=floodFreq.gte(1).fastDistanceTransform(256).sqrt().multiply(SCALE);
var absenceMask=domain.and(floodMask.not()).and(distFlood.gt(BUFFER_M));   // <<< buffer varies
var label=ee.Image(0).byte().rename('label').where(floodMask,1).updateMask(floodMask.or(absenceMask));

var stack=predictors.addBands(ee.Image.pixelLonLat()).addBands(floodFreq.rename('freq')).addBands(label);
var samples=stack.stratifiedSample({numPoints:N,classBand:'label',region:region,scale:SCALE,
  projection:PROJ,seed:SEED,geometries:true,tileScale:4});
print('buffer',BUFFER_M,'m | size',samples.size());

var cols=['elevation','slope','aspect','curvature','hand','twi','spi','dist_river','drain_density','soil','lulc','longitude','latitude','freq','label'];
Export.table.toDrive({collection:samples,description:'LPP_paper5_samples_'+SUFFIX,
  folder:'GEE_LPP_CMIP6',fileFormat:'CSV',selectors:cols});
// RUN with BUFFER_M=0 -> export; then change to 500 -> run -> export. Both land in GEE_LPP_CMIP6/.
