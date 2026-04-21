import pandas as pd
import numpy as np
import altair as alt
from geopy.distance import geodesic

def load_data():
    # 1. Food Security Data
    food_df = pd.read_csv("data/undernourishment_raw.csv")
    # Process the raw World Bank export
    year_cols = [c for c in food_df.columns if c.startswith('YR')]
    food_df['latest_undernourishment'] = food_df[year_cols].bfill(axis=1).iloc[:, 0]
    food_df = food_df[['Country', 'latest_undernourishment']]
    print(food_df.head())
    # 2. Major Ports Data
    ports_df = pd.read_csv("data/major_ports.csv")
    # 3. Country Centroids Data
    centroids_df = pd.read_csv("data/country_centroids.csv")

    return food_df, ports_df, centroids_df

def calculate_nearest_port_distance(country_df, ports_df):
    def get_min_dist(row):
        # Coordinates in country_centroids.csv are 'latitude' and 'longitude'
        country_coord = (row['latitude'], row['longitude'])
        # Coordinates in major_ports.csv are 'lat' and 'lon'
        distances = [geodesic(country_coord, (p['lat'], p['lon'])).km for _, p in ports_df.iterrows()]
        return min(distances)
    
    country_df['dist_to_nearest_port_km'] = country_df.apply(get_min_dist, axis=1)
    return country_df

def main():
    try:
        food_df, ports_df, centroids_df = load_data()
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the 'data' folder contains the required CSV files.")
        return

    # Merge food data with centroids
    # Note: Using 'left' merge to keep all food data, then dropping countries without coordinates
    merged_df = pd.merge(food_df, centroids_df, left_on='Country', right_on='name')
    
    # Calculate distances
    final_df = calculate_nearest_port_distance(merged_df, ports_df)
    
    # Visualization: Distance vs Undernourishment
    chart = alt.Chart(final_df).mark_point(filled=True, size=60).encode(
        x=alt.X('dist_to_nearest_port_km:Q', title='Distance to Nearest Major Port (km)'),
        y=alt.Y('latest_undernourishment:Q', title='Prevalence of Undernourishment (%)'),
        color=alt.Color('dist_to_nearest_port_km:Q', scale=alt.Scale(scheme='viridis')),
        tooltip=['Country', 'dist_to_nearest_port_km', 'latest_undernourishment']
    ).properties(
        width=600,
        height=400,
        title='Impact of Supply Chain Proximity on Food Security'
    ).interactive()

    # Show sample and save
    print("\nSample of Processed Data:")
    print(final_df[['Country', 'dist_to_nearest_port_km', 'latest_undernourishment']].head())
    
    # Save the final merged dataset for further analysis
    output_file = 'processed_food_security_analysis.csv'
    final_df.to_csv(output_file, index=False)
    print(f"\nAnalysis ready! Results saved to '{output_file}'")

if __name__ == "__main__":
    main()
