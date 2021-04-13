from esda.moran import Moran, Moran_Local
import geopandas as gpd
from geopandas import GeoDataFrame as gdf
import libpysal as lp
from libpysal import weights
from libpysal.weights import Queen, KNN, attach_islands
import mapclassify as mc
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import streamlit as st

@st.cache(allow_output_mutation=True)
def get_data():
    data = gpd.read_file('capstone1.shp')
    data = data.rename(columns = {'Municipali':'Municipal Code',
                                     'Municipa_1':'Municipality',
                                     'Mun_Povert':'Poverty Incidence',
                                     'beds_per_c': 'Health Facility Beds per 1000 people',
                                     'current_ca': 'Current New Cases',
                                     'total_inci': 'Total Cases per 100,000 people',
                                     'current_in': 'Current New Cases per 100,000 people',
                                     'case_fatal': 'Case Fatality Ratio',
                                     'population': 'No. People per 30 sq m',
                                     'elderly_po': 'No. of Elderly in Poverty',
                                     'total_case': 'Total Cases',
                                     'atrisk': 'Elderly Individuals at Risk',
                                     'total_beds': 'Total Beds'})
    #data['geometry'] = data['geometry'].to_crs(epsg = 3395)
    regions = data[['RegCode', 'Region', 'geometry']]
    regions = regions.dissolve(by='RegCode')
    province = data[['ProvinceCo', 'Province', 'geometry']]
    province = province.dissolve(by='ProvinceCo')
    return data, regions, province

def get_weights(df):
    wq = Queen.from_dataframe(df)
    wknn1 = KNN.from_dataframe(df)
    wfixed = weights.util.attach_islands(wq, wknn1)
    return wknn1


# compute the  local spatial autocorrelation
def compute_lsa(df, wq):
    variable = [ 'Poverty Incidence',
    'Health Facility Beds per 1000 people',
    'Current New Cases',
    'Total Cases per 100,000 people',
    'Current New Cases per 100,000 people',
    'Case Fatality Ratio',
    'No. People per 30 sq m',
    'No. of Elderly in Poverty',
    'Total Cases',
    'Elderly Individuals at Risk'
    ]
    lsa = [ 'LSA: Poverty Incidence',
    'LSA: Health Facility Beds per 1000 people',
    'LSA: Current New Cases',
    'LSA: Total Cases per 100,000 people',
    'LSA: Current New Cases per 100,000 people',
    'LSA: Case Fatality Ratio',
    'LSA: No. People per 30 sq m',
    'LSA: No. of Elderly in Poverty',
    'LSA: Total Cases',
    'LSA: Elderly Individuals at Risk'
    ]
    for i in range(len(variable)):
        lisa = Moran_Local(df[variable[i]], wq)
        sig = 1 * (lisa.p_sim < 0.05)
        hotspot = 1 * (sig * lisa.q==1)
        coldspot = 3 * (sig * lisa.q==3)
        doughnut = 2 * (sig * lisa.q==2)
        diamond = 4 * (sig * lisa.q==4)
        spots = hotspot + coldspot + doughnut + diamond
        spot_labels = [ '0 Not Significant', '1 Hot Spot', '2 Doughnut', '3 Cold spot', '4 Diamond']
        labels = [spot_labels[i] for i in spots]
        df[lsa[i]] = labels
    return df


@st.cache
def stdScaler(df):
    scaler = StandardScaler()
    return scaler.fit_transform(df)

@st.cache
def produceClusters(n_clusters, scaled_df, random_state=13):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(scaled_df)
    return kmeans.predict(scaled_df)

@st.cache
def kmeans_clustering(df):
    variable = [ 'Poverty Incidence',
    'Health Facility Beds per 1000 people',
    'Total Cases per 100,000 people',
    'Current New Cases per 100,000 people',
    'Case Fatality Ratio',
    'No. People per 30 sq m',
    'Elderly Individuals at Risk'
    ]
    df['Cluster Label'] = produceClusters(2, df[variable])
    table = df.groupby(['Cluster Label'])['Cluster Label'].count()
    if table[0]>table[1]:
        group1 = df[df['Cluster Label']==1]
        group1['Cluster Label'] = group1['Cluster Label'].replace(1, '1. Metropolis-like')
        other = df[df['Cluster Label']==0]
    else:
        group1 = df[df['Cluster Label']==0]
        group1['Cluster Label'] = group1['Cluster Label'].replace(0, '1. Metropolis-like')
        other = df[df['Cluster Label']==1]
    other['Cluster Label'] = produceClusters(2, other[variable])
    table = other.groupby(['Cluster Label'])['Cluster Label'].count()
    if table[0]>table[1]:
        group2 = other[other['Cluster Label']==1]
        group2['Cluster Label'] = group2['Cluster Label'].replace(1, '2. City-like')
        other = other[other['Cluster Label']==0]
    else:
        group2 = other[other['Cluster Label']==0]
        group2['Cluster Label'] = group2['Cluster Label'].replace(0, '2. City-like')
        other = other[other['Cluster Label']==1]
    other['Cluster Label'] = produceClusters(2, other[variable])
    table = other.groupby(['Cluster Label'])['Cluster Label'].count()
    if table[0]>table[1]:
        other['Cluster Label'] = other['Cluster Label'].replace(1, '3. Semiurban-like')
        other['Cluster Label'] = other['Cluster Label'].replace(0, '4. Rural-like')
    else:
        other['Cluster Label'] = other['Cluster Label'].replace(0, '3. Semiurban-like')
        other['Cluster Label'] = other['Cluster Label'].replace(1, '4. Rural-like')
    kmeans1 = pd.concat([group1, group2, other], ignore_index = True)
    return kmeans1

@st.cache
def get_custom_color_palette():
    return ListedColormap(['darkslategray', 'slategrey', 'darkgrey', 'gainsboro'])

@st.cache
def region_options():
    list = regions['Region'].unique()
    list = np.insert(list, 0, 'All Regions')
    return list

@st.cache
def province_options(region_choice):
    province  = original_df[original_df['Region']==region_choice]
    list = province['Province'].unique()
    list = np.insert(list, 0, 'All Provinces')
    return list

def get_coordinates(region_choice, province_choice):
    if (region_choice == 'All Regions') & (province_choice=='All Provinces'):
        coordinates = regions.geometry.total_bounds
    else:
        if province_choice =='All Provinces':
            coordinates = regions[regions['Region']==region_choice].geometry.total_bounds
        else:
            coordinates = province[province['Province']==province_choice].geometry.total_bounds
    return coordinates

def get_subsetdf(region_choice, province_choice, working_df):
    if (region_choice == 'All Regions') & (province_choice=='All Provinces'):
        subsetdf = working_df
    else:
        if province_choice =='All Provinces':
            subsetdf = working_df[working_df['Region']==region_choice]
        else:
            subsetdf = working_df[working_df['Province']==province_choice]
    return subsetdf

def get_graphs(coordinates, subsetdf, working_df):
    variable = [
            'Health Facility Beds per 1000 people',
            'Total Cases per 100,000 people',
            'Current New Cases per 100,000 people',
            #'Case Fatality Ratio',
            'No. People per 30 sq m',
            'Elderly Individuals at Risk',
            'Poverty Incidence'
            ]
    variable_choice = st.selectbox('Choose Variable', variable, 0)
    st.sidebar.header('Chosen Variable')
    st.sidebar.text(variable_choice)
    get_histogram(variable_choice, subsetdf)
    get_map(coordinates, working_df, variable_choice, subsetdf)
    return

@st.cache(suppress_st_warning=True)
def get_histogram(variable_choice, subsetdf):
    st.header('Histogram of the Variable')
    #minv, maxv = st.slider('Variable Range', subsetdf[variable_choice].min(), subsetdf[variable_choice].max(), (subsetdf[variable_choice].min(), subsetdf[variable_choice].max()))
    f = px.histogram(subsetdf, x = variable_choice, title = 'Variable Distribution')
    f.update_xaxes(title = variable_choice)
    f.update_yaxes(title = "No. of Municipalities")
    st.plotly_chart(f)
    return

@st.cache(suppress_st_warning=True)
def get_map(coordinates, working_df, variable_choice, subsetdf):
    st.header('Area Map')
    map = st.selectbox('Choose the Map', ['Chloropleth Map of Variable',
    'Local Spatial Autocorrelation Result',
    'KMeans Clustering Result'], 0 )
    if map == 'Chloropleth Map of Variable':
        st.text('Chloropleth map shows the distribution of our selected variable')
        chloropleth(coordinates, working_df, variable_choice, subsetdf)
    elif map=='Local Spatial Autocorrelation Result':
        st.text('Local Spatial Autocorrelation is an analysis to identify local clusters and local spatial outliers')
        st.text('Hot spot = area of significantly high values')
        st.text('Doughnut = area of low values surrounded by area of higher  values')
        st.text('Cold spot = area of significantly low values')
        st.text('Diamond = area of high values surrounded by area of lower  values')
        LSA(coordinates, working_df, variable_choice, subsetdf)
    else:
        st.text('Kmeans clustering is a machine learning algorithm to group together municipalities with similar characteristics')
        st.text('Metropolis-like = highest in terms of cases, population and resources')
        st.text('City-like = second highest in terms of cases, population and resources')
        st.text('Semiurban-like = low in terms of cases, population and resources ')
        st.text('Rural-like = lowest in terms of cases, population and resources')
        cluster(coordinates, working_df, variable_choice, subsetdf)
    return

@st.cache(suppress_st_warning=True)
def chloropleth(coordinates, working_df, variable_choice, subsetdf):
    fig, ax = plt.subplots(1, figsize=(12, 9))
    working_df.plot(column=variable_choice, cmap='Blues', linewidth=0.35, ax=ax,
    legend_kwds = {'loc' : 'upper left'}, edgecolor='0.9', scheme = 'Quantiles', k=5, legend = True, alpha = 0.6)
    province.boundary.plot(ax=ax, linewidth = 0.65, edgecolor='0.1')
    regions.boundary.plot(ax=ax, linewidth = 0.9, edgecolor='sandybrown')
    xlim = ([coordinates[0], coordinates[2]])
    ylim = ([coordinates[1], coordinates[3]])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis('off')
    plt.suptitle(variable_choice)
    st.pyplot(fig)
    variable = ['Province', 'Municipality', variable_choice]
    st.write(subsetdf[variable])
    return

@st.cache(suppress_st_warning=True)
def LSA(coordinates, working_df, variable_choice, subsetdf):
    lsa_choice = choice(variable_choice)
    hmap = ListedColormap([ 'grey', 'red', 'lightblue', 'blue', 'pink'])
    fig, ax = plt.subplots(1, figsize=(12, 9))
    working_df.plot(column=lsa_choice, cmap=hmap, categorical = True, linewidth=0.35, ax=ax,
    legend_kwds = {'loc' : 'upper left'}, edgecolor='0.9', k=2, legend = True, alpha = 0.6)
    province.boundary.plot(ax=ax, linewidth = 0.65, edgecolor='0.1')
    regions.boundary.plot(ax=ax, linewidth = 0.9, edgecolor='sandybrown')
    xlim = ([coordinates[0], coordinates[2]])
    ylim = ([coordinates[1], coordinates[3]])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.suptitle(lsa_choice)
    st.pyplot(fig)
    variable = ['Province', 'Municipality', lsa_choice]
    st.write(subsetdf[variable])
    return

@st.cache(suppress_st_warning=True)
def choice(variable_choice):
    lchoice = variable_choice
    if variable_choice == 'Poverty Incidence' :
        lchoice  = 'LSA: Poverty Incidence'
    elif variable_choice == 'Health Facility Beds per 1000 people':
        lchoice = 'LSA: Health Facility Beds per 1000 people'
    elif variable_choice == 'Total Cases per 100,000 people':
        lchoice = 'LSA: Total Cases per 100,000 people'
    elif variable_choice == 'Current New Cases per 100,000 people':
        lchoice = 'LSA: Current New Cases per 100,000 people'
    elif variable_choice == 'Case Fatality Ratio':
        lchoice = 'LSA: Case Fatality Ratio'
    elif variable_choice == 'LSA: No. People per 30 sq m':
        lchoice = 'LSA: No. People per 30 sq m'
    elif variable_choice == 'Elderly Individuals at Risk':
        lchoice = 'LSA: Elderly Individuals at Risk'
    return lchoice


@st.cache(suppress_st_warning=True)
def cluster(coordinates, working_df, variable_choice, subsetdf):
    fig, ax = plt.subplots(1, figsize=(12, 9))
    hmap = ListedColormap(['darkslategray', 'slategrey', 'darkgrey', 'gainsboro'])
    working_df.plot(column='Cluster Label', cmap=hmap, linewidth=0, ax=ax,
    #order = ['Metropolis-like', 'City-like', 'Semiurban-like', 'Rural-like'],
    edgecolor='0.9', legend_kwds = {'loc' : 'upper left'}, legend = True)
    province.boundary.plot(ax=ax, linewidth = 0.75, edgecolor='0.1')
    regions.boundary.plot(ax=ax, linewidth = 1, edgecolor='sandybrown')
    xlim = ([coordinates[0], coordinates[2]])
    ylim = ([coordinates[1], coordinates[3]])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis('off')
    st.pyplot(fig)
    variable = ['Province', 'Municipality', 'Cluster Label']
    st.write(subsetdf[variable])
    return

def update_data(subsetdf, working_df, original_df):
    st.header('Update Data')
    subsetdf = subsetdf.sort_values(by = ['Municipal Code'])
    municipal_code = st.selectbox('Choose the Municipal Code of the City/Municipality to be updated',
    subsetdf['Municipal Code'].unique())
    st.text("Reference for the Municipal Code")
    variable = [ 'Municipal Code',
        'Province',
        'Municipality',
        'Health Facility Beds per 1000 people'
        ]
    st.write(subsetdf[variable])
    st.sidebar.success(working_df[working_df['Municipal Code']==municipal_code][['Municipality', 'Province']])
    ab = working_df[working_df['Municipal Code']==municipal_code][['Health Facility Beds per 1000 people',
    'Total Cases per 100,000 people', 'Current New Cases per 100,000 people', 'Case Fatality Ratio',
    'No. People per 30 sq m', 'Elderly Individuals at Risk', 'Poverty Incidence']]
    hf = st.number_input('Health Facility Beds per 1000 people', min_value = 0.0,
    value = float(ab['Health Facility Beds per 1000 people']))
    tc = st.number_input('Total Cases per 100,000 people', min_value = 0.0,
    value = float(ab['Total Cases per 100,000 people']))
    cc = st.number_input('Current New Cases per 100,000 people', min_value = 0.0,
    value = float(ab['Current New Cases per 100,000 people']))
    cr = st.number_input('Case Fatality Ratio', min_value = 0.0,
    value = float(ab['Case Fatality Ratio']))
    no = st.number_input('No. People per 30 sq m', min_value = 0.0,
    value = float(ab['No. People per 30 sq m']))
    er = st.number_input('Elderly Individuals at Risk', min_value = 0.0,
    value = float(ab['Elderly Individuals at Risk']))
    pi = st.number_input('Poverty Incidence', min_value = 0.0,
    value = float(ab['Poverty Incidence']))
    if (st.button('Click to Update Data')):
        working_df = update(working_df, municipal_code, hf, tc, cc, cr, no, er, cc)
        st.success('Data updated')
    if (st.button('Click to Update Model')):
        working_df = compute_lsa(working_df, wq)
        working_df = kmeans_clustering(working_df)
        st.success('Model updated')
    if (st.button('Click to Reset Changes')):
        working_df = original_df.copy()
    return working_df, municipal_code

@st.cache(suppress_st_warning=True)
def update(working_df, municipal_code, hf, tc, cc, cr, no, er, pi):
    df = working_df.set_index('Municipal Code')
    df.at[municipal_code, 'Health Facility Beds per 1000 people'] = hf
    df.at[municipal_code, 'Total Cases per 100,000 people'] = tc
    df.at[municipal_code, 'Current New Cases per 100,000 people'] = cc
    df.at[municipal_code, 'Case Fatality Ratio'] = cr
    df.at[municipal_code, 'No. People per 30 sq m'] = no
    df.at[municipal_code, 'Elderly Individuals at Risk'] = er
    df.at[municipal_code, 'Poverty Incidence'] = pi
    working_df = df.reset_index()
    return working_df

@st.cache(suppress_st_warning=True)
def get_info():
    st.header("Variables Used in The Model")
    st.markdown("The following are the variables used in the classification model")
    st.markdown("> Health Facility Beds per 1000 people - this is the ratio of beds of infirmaries and \
    hospitals within a city/municipality to their population.")
    st.markdown("> Total Cases per 100,000 people - ratio of confirmed Covid 19 cases per city \
    /municipality to the population. The total cases as of March 17, 2021 were used for the initial \
    clustering model.")
    st.markdown("> Current New Cases per 100,000 people - ration of total confirmed Covid 19 cases \
    for the past 2 weeks and the population. The new cases that was used for the model are the published\
    cases wherein their date of onset of symptoms/ date of specimen collection/ date of confirmation \
    (whichever is available) falls within March 3 -17, 2021")
    st.markdown('> Case Fatality Ratio - computed by dividing the total covid deaths by the total covid \
    cases by city/municipality')
    st.markdown("> Poverty Incidence - percent of the city/municipality's population that are below \
    the povery line")
    st.markdown("> No. People per 30 sq m - estimated number and a measure of population density")
    st.markdown("> Elderly Individuals at Risk - estimated number of elderly individuals that are \
    at risk of contracting COVID 19. It was assumed that those that already had COVID-19 had some \
    immunity to the disease thus this varible was computed by subtracting the estimated number of \
    elderly individuals minus the total number of COVID 19 cases in the elderly ")
    return

original_df, regions, province = get_data()
wq = get_weights(original_df)
original_df = compute_lsa(original_df, wq)
original_df = kmeans_clustering(original_df)
working_df = original_df
st.title("Clustering Senior Citizen's Vaccine Distribution Prioritization in the Philippines")
st.sidebar.header("User Selection")
region_choice = st.sidebar.selectbox('Region', region_options(), 0)
province_choice = st.sidebar.selectbox('Province',
    province_options(region_choice), 0)
coordinates = get_coordinates(region_choice, province_choice)
subsetdf = get_subsetdf(region_choice, province_choice, working_df)
#working_df = update_data(subsetdf, working_df, original_df)
#if (st.sidebar.button('Click for Graphs')):
get_graphs(coordinates, subsetdf, working_df)
if (st.sidebar.button('Click for Info')):
    get_info()
