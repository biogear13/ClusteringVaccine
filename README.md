# COVID-19 Vaccine Prioritisation. Clustering Philippine Municipalities by Risk

**Generoso Roberto | Eskwelabs Data Science Fellowship Capstone | 2021**

---

## The Problem

In early 2021, the Philippines began COVID-19 vaccination with a limited supply and 1,480 municipalities to prioritise. The Department of Health needed a transparent, data-driven method to decide which communities should receive vaccines first. The challenge was not a shortage of data. it was that the available data came from multiple government agencies in inconsistent formats, covered different geographic levels, and had never been combined into a single analytical framework for this purpose.

This project built that framework.

---

## What This Project Does

This project uses k-means clustering and geospatial analysis to group all 1,480 Philippine municipalities into four priority tiers for COVID-19 vaccination, based on publicly available government data.

A Streamlit application allows public health decision-makers to:

- View choropleth maps of any risk variable across the Philippines, filterable by region and province
- Explore local spatial autocorrelation results (hot spots, cold spots, spatial outliers) for each variable
- View the four-cluster vaccination priority map at national, regional, or provincial level
- Update individual municipality data and rerun the model to see how priority assignments change

---

## Data Sources

Five government datasets were collected, cleaned, and merged at the municipality level:

| Dataset | Source | Coverage |
|---|---|---|
| Administrative boundaries and municipality codes | PhilGIS (Philippine GIS Data Clearinghouse) | All 1,480 municipalities |
| Poverty incidence by municipality | Philippine Statistics Authority (PSA) | All municipalities |
| Health facility beds by municipality | Department of Health (DOH) facility registry | All municipalities |
| COVID-19 confirmed cases and deaths | DOH COVID-19 tracker | Up to March 17, 2021 |
| Population estimates and elderly population | Philippine Statistics Authority | All municipalities |

Merging these datasets required resolving inconsistent municipality naming conventions across agencies and handling missing values for 94 municipalities with no recorded health facility beds.

---

## Variables Used in the Clustering Model

Seven variables were selected based on public health relevance and data availability:

- **Health facility beds per 1,000 people**: proxy for local healthcare capacity
- **Total COVID-19 cases per 100,000 people**: cumulative disease burden
- **Current new cases per 100,000 people**: active transmission pressure
- **Case fatality ratio**: local severity indicator
- **Population density (persons per 30 sq m)**: transmission risk amplifier
- **Elderly individuals at risk**: vulnerable population estimate (total elderly minus confirmed cases, assuming prior infection confers some immunity)
- **Poverty incidence**: access barrier to healthcare

All variables were standardised using StandardScaler before clustering.

---

## Methods

### Spatial Analysis

Before clustering, global and local spatial autocorrelation were computed for each variable using Moran's I and Local Indicators of Spatial Association (LISA). This confirmed that the variables are not randomly distributed across space. high-risk municipalities cluster together geographically, which validates the use of spatial data in prioritisation.

Queen contiguity weights were used to define spatial neighbours, with KNN attachment for island municipalities that have no contiguous neighbours.

### Clustering Approach

The optimal number of clusters was determined using the elbow method and silhouette scores. Rather than a single k-means run, a hierarchical k-means approach was used:

1. All municipalities are split into two groups: the largest (most populous and resource-intensive) group is labelled **Metropolis-like**
2. The remaining municipalities are split again: the larger group becomes **City-like**
3. The remainder is split a final time into **Semiurban-like** and **Rural-like**

This approach produces more interpretable and actionable clusters than a single four-cluster run, because it separates the extreme outliers (Metro Manila and major cities) before clustering the majority of municipalities.

### Four Priority Clusters

| Cluster | Characteristics | Vaccination priority rationale |
|---|---|---|
| Metropolis-like | Highest case burden, highest population density, most health facility beds | High absolute case numbers demand early allocation despite relatively better infrastructure |
| City-like | Elevated cases and density, moderate health capacity | Second priority: significant transmission risk with constrained resources |
| Semiurban-like | Low-to-moderate cases, limited health capacity | Third priority: lower current burden but high vulnerability if cases increase |
| Rural-like | Lowest cases, lowest density, fewest resources | Lowest immediate risk but highest equity concern: lowest access to care |

---

## Tools and Libraries

| Purpose | Tool |
|---|---|
| Data processing and merging | Python, Pandas, NumPy |
| Geospatial analysis | GeoPandas, Shapely |
| Spatial autocorrelation | libpysal, esda (Moran's I, LISA) |
| Clustering | scikit-learn (KMeans) |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Interactive application | Streamlit |

---

## Repository Structure

```
ClusteringVaccine/
├── Capstone.ipynb          # Full analysis notebook: EDA, spatial analysis, clustering
├── dataextraction.py       # Data pipeline: merges five government datasets into one geodataframe
├── streamlit.py            # Interactive Streamlit application
├── requirements.txt        # Python dependencies
├── setup.sh                # Streamlit server configuration
└── README.md
```

The Streamlit application reads from `capstone1.shp`, a processed shapefile produced by `dataextraction.py` after merging and cleaning all five source datasets.

---

## Key Findings

- 27 municipalities had no health facility beds at all, including several with active COVID-19 cases: these represent the most critical equity gaps in the analysis
- Spatial autocorrelation confirmed significant clustering of high-risk municipalities in Metro Manila and surrounding provinces, consistent with known epidemiological patterns
- The Metropolis-like cluster (Metro Manila and major cities) accounted for a disproportionate share of both cases and elderly individuals at risk, supporting early prioritisation of urban centres
- The Rural-like cluster had the lowest case burden but also the lowest healthcare access, representing an equity argument for parallel allocation even during early rollout

---

## Limitations

- Data is cross-sectional as of March 2021 and reflects early-pandemic conditions before Delta and Omicron variants
- Poverty incidence data is from 2018 PSA estimates, not contemporaneous with COVID data
- The model clusters based on available indicators: it does not account for cold chain capacity, local government implementation capacity, or vaccine hesitancy
- Not an official DOH tool and was not used in actual policy decisions
- Clustering is descriptive and deterministic given fixed random state: different random seeds may produce slightly different cluster assignments

---

## About the Author

Generoso Roberto is a GMC-registered doctor and public health professional based in the UK, working at the intersection of digital health, data analytics, and health equity. This project was completed as the capstone for the Eskwelabs Data Science Fellowship in 2021.

[LinkedIn](https://www.linkedin.com/in/generoso-roberto) | [GitHub](https://github.com/biogear13)
