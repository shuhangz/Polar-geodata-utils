# Sea ice drift Correction Tool

## Overview
This Python script is designed to perform sea ice drift correction for images captured by DJI drones based on their EXIF data and a reference GPS database (from the research vessel). It extracts geographical coordinates, time information, and other metadata from the images, and then uses this data to correct any drift in the images' geolocation.

## Requirements
- Python 3.x
- Pillow (PIL)
- NumPy
- osgeo (GDAL/OGR)
- tqdm
- sqlite3

## Installation
To run this script, ensure you have all the required packages installed. You can install them using pip:

```bash
pip install Pillow numpy osgeo tqdm
```

## Usage
To use this script, you need to have a GPS database in SQLite format and a directory containing the images to be corrected. Run the script by executing:

```bash
python drift_correction.py
```

Make sure to replace the `path_gps_database` and `path_to_search` variables with the actual paths to your database and image directory.

## Code Explanation

### Functions

- `find_files(path, tp)`: Searches for all files with the specified extension in the given directory and its subdirectories.

- `convert_to_degrees(value)`: Converts degrees, minutes, and seconds into decimal degrees.

- `get_exif_data(image_path)`: Retrieves EXIF data from an image.

- `parse_exif(exif_data)`: Extracts time, longitude, latitude, and altitude from the EXIF data.

- `query_specific_time(path, year, month, day, hour, minute, second)`: Queries the GPS database for latitude and longitude at a specific time.

- `DriftCorr(ref_lat, ref_lon, loc_gps_lat, loc_gps_lon, loc_img_lat, loc_img_lon)`: Performs the drift correction by transforming coordinates and adjusting for differences.

### Main Logic

The `main()` function orchestrates the process:
1. Finds all .JPG files in the specified directory.
2. Extracts EXIF data and parses it to get the necessary information.
3. Queries the GPS database for reference coordinates at the time the image was taken.
4. Corrects the drift by comparing the image coordinates with the reference coordinates.
5. Outputs the corrected coordinates.

## License
This project is open-source and available under the MIT License.

## Contact
For any questions or issues, please contact Jichang Shen at  13669549372@163.com.