import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import geopandas as gpd
from geopandas import GeoDataFrame as gdf
from shapely.geometry import Point
import seaborn as sns
import libpysal as lp
from libpysal.weights import Queen, KNN, attach_islands
from libpysal import weights
import mapclassify as mc
from esda.moran import Moran, Moran_Local
from sklearn.cluster import KMeans, AgglomerativeClustering
def get_PSGC():
    regionPSGC_df = pd.read_csv('PSGC/regionPSGC.csv', encoding = 'unicode_escape')
    regionPSGC_df.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"],
                          value=["", ""], regex=True, inplace=True)
    provPSGC_df = pd.read_csv('PSGC/provPSGC.csv', encoding = 'unicode_escape')
    provPSGC_df.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"],
                        value=["", ""], regex=True, inplace=True)
    munPSGC_df = pd.read_csv('PSGC/munPSGC.csv', encoding = 'unicode_escape')
    munPSGC_df.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"],
                       value=["", ""], regex=True, inplace=True)
    #barangayPSGC_df = pd.read_csv('PSGC/barangayPSGC.csv', encoding = 'unicode_escape')
    #barangayPSGC_df.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"],
    #                        value=["", ""], regex=True, inplace=True)
    #munPSGC_df = munPSGC_df.merge(barangayPSGC_df, how = 'right', on = 'MC')
    provPSGC_df = provPSGC_df.merge(munPSGC_df, how = 'right', on = 'PC')
    regionPSGC_df = regionPSGC_df.merge(provPSGC_df, how = 'right', on = 'RC')
    #regionPSGC_df = regionPSGC_df.drop(columns=['Old names_x','Old names_y'])
    regionPSGC_df = regionPSGC_df.loc[:, ~regionPSGC_df.columns.str.contains('^Unnamed')]
    return regionPSGC_df

def get_shapefile():
    shapefile = gpd.read_file('MuniCities/Municipalities.shp')
    shapefile['ADM3_PCODE'] = shapefile['ADM3_PCODE'].apply(lambda x: x.replace('PH',''))
    shapefile['ADM3_PCODE'] = shapefile['ADM3_PCODE'].astype('int64')
    shapefile['ADM2_PCODE'] = shapefile['ADM2_PCODE'].apply(lambda x: x.replace('PH',''))
    shapefile['ADM2_PCODE'] = shapefile['ADM2_PCODE'].astype('int64')
    shapefile['ADM1_PCODE'] = shapefile['ADM1_PCODE'].apply(lambda x: x.replace('PH',''))
    shapefile['ADM1_PCODE'] = shapefile['ADM1_PCODE'].astype('int64')
    shapefile = shapefile[['Shape_Leng', 'Shape_Area', 'ADM3_PCODE', 'geometry']]
    shapefile = shapefile.rename(columns = {'ADM3_PCODE':'MunicipalityCode'})
    return shapefile

def get_poverty():
    poverty_df = pd.read_csv('Poverty/poverty.csv', encoding = 'unicode_escape')
    #poverty_df.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"],
    #                      value=["", ""], regex=True, inplace=True)
    return poverty_df

def merge_file(shapefile, psgc, poverty):
    shapefile = shapefile.merge(psgc, how = 'left', on = 'MunicipalityCode')
    shapefile = shapefile.merge(poverty, how = 'left', on = 'MC')
    return shapefile

def get_DOH():
    # Get DOH database
    region = ['ARMM', 'CAR', 'Metromanila', 'Region1',
              'Region2', 'Region3', 'Region4-a', 'Region4-b',
              'Region5', 'Region6', 'Region7', 'Region8',
              'Region9', 'Region10', 'Region11', 'Region12', 'Region13']

    hospital = []

    for KEYWORD in region:
        region_df = pd.read_csv('DOH/'+KEYWORD+'.csv', index_col=0, encoding = 'unicode_escape')
        region_df.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"],
                          value=["", ""], regex=True, inplace=True)
        # print(KEYWORD+': %s' % region_df.shape[0])
        region_df = region_df.dropna(axis=0, subset=['Facility Name'])
        # print(KEYWORD+' where null values where dropped: %s' \
        #        % region_df.shape[0])
        hospital.append(region_df)

    hospital_df = pd.concat(hospital)
    hospital_df = hospital_df[['Health Facility Code Short',
                             'Facility Name',
                             'Health Facility Type',
                              'Ownership Major Classification',
                              'City/Municipality PSGC',
                              'Hospital Licensing Status',
                              'Service Capability',
                              'Bed Capacity'
                             ]]
    hospital_df = hospital_df.rename(columns = {'City/Municipality PSGC':'MunicipalityCode'})
    inpatient = ['Hospital', 'Infirmary']
    hospital_df = hospital_df[hospital_df['Health Facility Type'].isin(inpatient)]
    #looking at the data the null values in the Service Capacity are for the Infirmary since Service Capability is for hospitals
    hospital_df['Service Capability'] = hospital_df['Service Capability'].fillna('Infirmary')
    servicelevel = ['Level 1', 'Infirmary', 'Level 2', 'Level 3']
    hospital_df = hospital_df[hospital_df['Service Capability'].isin(servicelevel)]
    return hospital_df

def get_covid():
    covid_df = pd.read_csv('covid/covid.csv', encoding = 'unicode_escape')
    covid_df.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"],
                          value=["", ""], regex=True, inplace=True)
    covid_df = covid_df.rename(columns = {'CityMuniPSGC':'MunicipalityCode'})
    return covid_df
#create a data frame for the different datasets
hospital_df = get_DOH()
hospital_df.info()
psgc_df = get_PSGC()
psgc_df.info()
poverty_df = get_poverty()
poverty_df.info()
shapefile = get_shapefile()
#merge the different datframes that coould be combined
final_df = merge_file(shapefile, psgc_df, poverty_df)
final_df.info()
#seperate the the hospital dataframe by seperate service capability
infirmary = hospital_df[hospital_df['Service Capability']=='Infirmary']
level1 = hospital_df[hospital_df['Service Capability']=='Level 1']
level2 = hospital_df[hospital_df['Service Capability']=='Level 2']
level3 = hospital_df[hospital_df['Service Capability']=='Level 3']
#get the rows that has missing bed capacity
missing_infirmary = infirmary[infirmary.isnull().any(axis=1)]
missing_level1 = level1[level1.isnull().any(axis=1)]
missing_level2 = level2[level2.isnull().any(axis=1)]
missing_level3 = level3[level3.isnull().any(axis=1)]
missing_infirmary = missing_infirmary.drop(columns = 'Bed Capacity')
missing_level1 = missing_level1.drop(columns = 'Bed Capacity')
missing_level2 = missing_level2.drop(columns = 'Bed Capacity')
missing_level3 = missing_level3.drop(columns = 'Bed Capacity')
infirmary = infirmary[~infirmary.isnull().any(axis=1)]
level1 = level1[~level1.isnull().any(axis=1)]
level2 = level2[~level2.isnull().any(axis=1)]
level3 = level3[~level3.isnull().any(axis=1)]
missing_infirmary.info()
missing_level1.info()
missing_level2.info()
missing_level3.info()
#for the missing values the dataset will be used to supplement.
infirmaryph = pd.read_csv('PhilHealth/infirmary.csv', encoding = 'unicode_escape')
infirmaryph.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"],
                          value=["", ""], regex=True, inplace=True)
infirmaryph = infirmaryph[['NAME OF INSTITUTION', 'ABC']]
infirmaryph = infirmaryph.rename(columns = {'NAME OF INSTITUTION':'Facility Name',
                                           'ABC': 'Bed Capacity'})
missing_infirmary = missing_infirmary.merge(infirmaryph, how = 'left', on = 'Facility Name')
missing_infirmary = missing_infirmary.dropna()
missing_infirmary.info()
infirmary = pd.concat([infirmary, missing_infirmary])
infirmary.info()
level1ph = pd.read_csv('PhilHealth/level1.csv', encoding = 'unicode_escape')
level1ph.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"],
                          value=["", ""], regex=True, inplace=True)
level1ph = level1ph[['NAME OF INSTITUTION', 'ABC']]
level1ph = level1ph.rename(columns = {'NAME OF INSTITUTION':'Facility Name',
                                           'ABC': 'Bed Capacity'})
missing_level1 = missing_level1.merge(level1ph, how = 'left', on = 'Facility Name')
missing_level1 = missing_level1.dropna()
missing_level1.info()
level1 = pd.concat([level1, missing_level1])
level1.info()
level2ph = pd.read_csv('PhilHealth/level2.csv', encoding = 'unicode_escape')
level2ph.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"],
                          value=["", ""], regex=True, inplace=True)
level2ph = level2ph[['NAME OF INSTITUTION', 'ABC']]
level2ph = level2ph.rename(columns = {'NAME OF INSTITUTION':'Facility Name',
                                           'ABC': 'Bed Capacity'})
missing_level2 = missing_level2.merge(level2ph, how = 'left', on = 'Facility Name')
missing_level2 = missing_level2.dropna()
missing_level2.info()
level2 = pd.concat([level2, missing_level2])
level2.info()
level3ph = pd.read_csv('PhilHealth/level3.csv', encoding = 'unicode_escape')
level3ph.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"],
                          value=["", ""], regex=True, inplace=True)
level3ph = level3ph[['NAME OF INSTITUTION', 'ABC']]
level3ph = level3ph.rename(columns = {'NAME OF INSTITUTION':'Facility Name',
                                           'ABC': 'Bed Capacity'})
missing_level3 = missing_level3.merge(level3ph, how = 'left', on = 'Facility Name')
missing_level3 = missing_level3.dropna()
missing_level3.info()
level3 = pd.concat([level3, missing_level3])
level3.info()
#aggregate by number of hospital per municipality and number of beds per municpality per service level
inf = infirmary.groupby(['MunicipalityCode']).agg(infirmary_num = ('Bed Capacity','count'),
                                            infirmary_beds = ('Bed Capacity',
                                                                'sum')).reset_index()
lvl1 = level1.groupby(['MunicipalityCode']).agg(level1_num = ('Bed Capacity','count'),
                                            level1_beds = ('Bed Capacity',
                                                                'sum')).reset_index()
lvl2 = level2.groupby(['MunicipalityCode']).agg(level2_num = ('Bed Capacity','count'),
                                            level2_beds = ('Bed Capacity',
                                                                'sum')).reset_index()
lvl3 = level3.groupby(['MunicipalityCode']).agg(level3_num = ('Bed Capacity','count'),
                                            level3_beds = ('Bed Capacity',
                                                                'sum')).reset_index()
#merge the hospital data to the final data frame
final_df = final_df.merge(inf, how = 'left', on = 'MunicipalityCode')
final_df = final_df.merge(lvl1, how = 'left', on = 'MunicipalityCode')
final_df = final_df.merge(lvl2, how = 'left', on = 'MunicipalityCode')
final_df = final_df.merge(lvl3, how = 'left', on = 'MunicipalityCode')
final_df.info()
covid_df['DateRepConf']= pd.to_datetime(covid_df['DateRepConf'])
covid_df['DateResultRelease']= pd.to_datetime(covid_df['DateResultRelease'])
covid_df['DateSpecimen']= pd.to_datetime(covid_df['DateSpecimen'])
covid_df['DateOnset']= pd.to_datetime(covid_df['DateOnset'])
#fillin the missing dates
covid_df['DateResultRelease'] = covid_df['DateResultRelease'].fillna(covid_df['DateRepConf'])
covid_df['DateSpecimen'] = covid_df['DateSpecimen'].fillna(covid_df['DateResultRelease'])
covid_df['DateOnset'] = covid_df['DateOnset'].fillna(covid_df['DateSpecimen'])
covid_df.info()

#get only the cases that  until 2021-03-17
cutoff = '2021-03-17'
covid_df = covid_df.set_index('DateOnset')
covid_df = covid_df.sort_index(ascending = False)
covid_df = covid_df[cutoff:]
covid_df.info()

#remove the rows with no municipality data
covid_df = covid_df[~covid_df['MunicipalityCode'].isna()]
covid_df.info()

#get the total number of infected individuals per municipality.
covidcases = covid_df.groupby(['MunicipalityCode']).agg(total_cases = ('CaseCode','count')).reset_index()
covidcases.describe()

#get the total number of senior citizens that was infected with covid
seniorcases = covid_df[covid_df['Age']>=60]
seniorcases = seniorcases.groupby(['MunicipalityCode']).agg(seniorcitizen_cases = ('CaseCode','count')).reset_index()
seniorcases.describe()

#get the total number of covid deaths
mortality = covid_df[covid_df['RemovalType'].str.contains("DIED", na = True)]
mortality = mortality.groupby(['MunicipalityCode']).agg(num_of_deaths = ('CaseCode','count')).reset_index()
mortality['num_of_deaths'].sum()

#we would define current infected by those that are are have 14 days or less since the time that the data was extracted (March15)
new = '2021-03-03'
currentcases = covid_df[:new]
current = currentcases.groupby(['MunicipalityCode']).agg(current_cases = ('CaseCode','count')).reset_index()
current['current_cases'].sum()

#attach the covid data to the final_df
final_df = final_df.merge(covidcases, how = 'left', on = 'MunicipalityCode')
final_df = final_df.merge(seniorcases, how = 'left', on = 'MunicipalityCode')
final_df = final_df.merge(mortality, how = 'left', on = 'MunicipalityCode')
final_df = final_df.merge(current, how = 'left', on = 'MunicipalityCode')
final_df.info()

def get_population():
    popdensity = pd.read_csv('Population/popdensity.csv', encoding = 'unicode_escape')
    elderly = pd.read_csv('population/elderly.csv', encoding = 'unicode_escape')
    popdensity = popdensity.merge(elderly, how = 'outer', on = ['longitude', 'latitude'])
    return popdensity

popdensity = pd.read_csv('Population/popdensity.csv', encoding = 'unicode_escape')
popdensity = popdensity.drop(columns = 'population_2015')
popdensity= gdf(popdensity, crs={'init':'epsg:4326'}, geometry=gpd.points_from_xy(popdensity.longitude, popdensity.latitude))
popdensity

popdensity = popdensity.to_crs('EPSG:4326')
density = gpd.sjoin(popdensity, shapefile, op='within')
density = density.groupby(['MunicipalityCode']).agg(population_density = ('population_2020','median'), tot_pop = ('population_2020', 'sum')).reset_index()
density = density.round({'population_density': 2, 'tot_pop': 0})

elderly = pd.read_csv('Population/elderly.csv', encoding = 'unicode_escape')
elderly= gdf(elderly, crs={'init':'epsg:4326'}, geometry=gpd.points_from_xy(elderly.longitude, elderly.latitude))
elderly = elderly.to_crs('EPSG:4326')
elder = gpd.sjoin(elderly, shapefile, op='within')

elderly = pd.read_csv('Population/elderly.csv', encoding = 'unicode_escape')
elderly= gdf(elderly, crs={'init':'epsg:4326'}, geometry=gpd.points_from_xy(elderly.longitude, elderly.latitude))
elderly = elderly.to_crs('EPSG:4326')
elder = gpd.sjoin(elderly, shapefile, op='within')

final_df = final_df.merge(density, how = 'left', on = 'MunicipalityCode')
final_df = final_df.merge(elder, how = 'left', on = 'MunicipalityCode')
final_df.info()

final_df = final_df.merge(density, how = 'left', on = 'MunicipalityCode')
final_df = final_df.merge(elder, how = 'left', on = 'MunicipalityCode')
final_df.info()

final_df['tot_pop'] = final_df['tot_pop'].fillna(final_df['MunPop'])
final_df.info()
final_df.fillna(0)

#Feature engineering
final_df['total_beds'] = final_df['infirmary_beds']+final_df['level1_beds']+final_df['level2_beds']+final_df['level3_beds']
final_df['beds_per_capita'] = final_df['total_beds']/final_df['tot_pop']*1000
final_df['total_incidence'] = final_df['total_cases']/final_df['tot_pop']*100000
final_df['current_incidence'] = final_df['current_cases']/final_df['tot_pop']*100000
final_df['case_fatality_rate'] = final_df.apply(lambda x : 0 if x['total_cases'] ==0 else (x['num_of_deaths']/x['total_cases']*100), axis = 1)
final_df['atrisk'] = final_df['elderly_population']-final_df['seniorcitizen_cases']
final_df['elderly_poor'] = final_df['elderly_population']*final_df['Mun_Poverty_Incidence']/100

final_df  = final_df[['Shape_Leng',
                                'Shape_Area',
                                'MunicipalityCode',
                                'geometry',
                                'RegCode',
                                'Region',
                                'ProvinceCode',
                                'Province',
                                'Mun_Poverty_Incidence',
                                'total_beds',
                                'beds_per_capita',
                                'total_cases',
                                'current_cases',
                                'total_incidence',
                                'current_incidence',
                                'case_fatality_rate',
                                'atrisk',
                                'elderly_poor']]

    final_df.to_file('capstone.shp')
