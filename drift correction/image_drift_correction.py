import os
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import numpy as np
from datetime import datetime, timedelta
from osgeo import ogr,osr
from tqdm import tqdm
import sqlite3

####-----------------------------------Batch reading JPG
def find_files(path,tp):
  
    jpg_files = []
    
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith(tp):
                abs_path = os.path.abspath(os.path.join(root, file))
                jpg_files.append(abs_path)
    
    return jpg_files

def convert_to_degrees(value):
    #Convert degrees, minutes, and seconds into a decimal degrees
    d = value[0] / 1
    m = value[1] / 60
    s = value[2] / 3600
    return round(d + m + s, 8)

def get_exif_data(image_path):
    #Retrieve EXIF data of images
    image = Image.open(image_path)
    exif_data = image._getexif()
    return exif_data

def parse_exif(exif_data):
    #Analyze EXIF data to extract time, longitude, latitude, and altitude
    lat = None
    lon = None
    alt = None
    yr, mnt, day, hr, mn, sc = None, None, None, None, None, None

    if exif_data:
        for tag, value in exif_data.items():
            decoded = TAGS.get(tag, tag)
            if decoded == 'DateTimeOriginal':
                yr, mnt, day = int(value[0:4]), int(value[5:7]), int(value[8:10])
                hr, mn, sc = int(value[11:13]), int(value[14:16]), int(value[17:19])
            elif decoded == 'GPSInfo':
                gps_data = exif_data[tag]
                for key in gps_data.keys():
                    sub_decoded = GPSTAGS.get(key, key)
                    if sub_decoded == 'GPSLatitude':
                        lat = gps_data[key]
                    elif sub_decoded == 'GPSLongitude':
                        lon = gps_data[key]
                    elif sub_decoded == 'GPSAltitude':
                        alt = gps_data[key]

        if lat and lon:
            lat = convert_to_degrees(lat)
            lon = convert_to_degrees(lon)

    lat = np.float32(lat) if lat is not None else None
    lon = np.float32(lon) if lon is not None else None
    alt = np.float32(alt) if alt is not None else None
    
    return lat, lon, alt, yr, mnt, day, hr, mn, sc



#-----------------------Define functions for accessing GPS database connections
#Retrieve latitude and longitude for a specific time point
def query_specific_time(path, year, month, day, hour, minute, second):
    year = int(float(year))
    month = int(float(month))
    day = int(float(day))
    hour = int(float(hour))
    minute = int(float(minute))
    second = int(float(second))
    
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT longitude, latitude FROM location_data WHERE
    year=? AND month=? AND day=? AND hour=? AND minute=? AND second=?
    ''', (year, month, day, hour, minute, second))
    result = cursor.fetchone()
    
    return result

#------------------------------------------------------Define the function for drift correction
def DriftCorr(ref_lat, ref_lon, loc_gps_lat, loc_gps_lon, loc_img_lat, loc_img_lon):
    
    ####Transform coordinates between lat lon and polar stereo (ps)
    srs_in = osr.SpatialReference()
    srs_in.ImportFromEPSG(4326)
    srs_out = osr.SpatialReference()
    srs_out.ImportFromEPSG(3413)
    ct = osr.CoordinateTransformation(srs_in,srs_out)

    #####Transform coordinates between polar stereo (ps) and lat lon
    srs_in_back = osr.SpatialReference()
    srs_in_back.ImportFromEPSG(3413)
    srs_out_back = osr.SpatialReference()
    srs_out_back.ImportFromEPSG(4326)
    ct_back = osr.CoordinateTransformation(srs_in_back,srs_out_back)
    
    ref_psx, ref_psy, ref_psz = ct.TransformPoint(ref_lat, ref_lon)
    loc_gps_psx, loc_gps_psy, loc_gps_psz = ct.TransformPoint(loc_gps_lat, loc_gps_lon)
    loc_img_psx, loc_img_psy, loc_img_psz = ct.TransformPoint(loc_img_lat, loc_img_lon)
    dx = ref_psx - loc_gps_psx
    dy = ref_psy - loc_gps_psy
    
    loc_img_psx_cor = loc_img_psx + dx
    loc_img_psy_cor = loc_img_psy + dy
    
    loc_img_lat_cor, loc_img_lon_cor, _ = ct_back.TransformPoint(loc_img_psx_cor, loc_img_psy_cor)
    
    return loc_img_lat_cor, loc_img_lon_cor


def main():
    path_gps_database=r'time_location.db'
    #Path of jpg file
    path_to_search = r'D:\Working_Project\Arctic_2024_Shuhang\Data\20240907_临时冰站_M2ET_热红外_processing\可见光'
    tp='.jpg'
    jpg_files = find_files(path_to_search,tp)
    
    file_names = np.array([])
    lat=np.array([])
    lon=np.array([])
    alt=np.array([])
    yr=np.array([])
    mnt=np.array([])
    day=np.array([])
    hr=np.array([])
    mn=np.array([])
    sc=np.array([])
    
    #The coefficients are latitude, longitude, altitude, year, month, day, hour, minute, and second, respectively
    for i in range(len(jpg_files)):
        
        image_path = jpg_files[i]
        exif_data = get_exif_data(image_path)
        a1, a2, a3, a4, a5, a6, a7, a8, a9 = parse_exif(exif_data)
        
        #Convert to UTC time 
        dt0=datetime(a4,a5,a6,a7,a8,a9)
        eight_hours = timedelta(hours=8)
        dt1 = dt0 - eight_hours
        
        a4=dt1.year
        a5=dt1.month
        a6=dt1.day
        a7=dt1.hour
        a8=dt1.minute
        a9=dt1.second
        
        # get image file name without path_gps_database
        image_file_name = os.path.basename(image_path)
        file_names=np.append(file_names,image_file_name)
        lat=np.append(lat,a1)
        lon=np.append(lon,a2)
        alt=np.append(alt,a3)
        yr=np.append(yr,a4)
        mnt=np.append(mnt,a5)
        day=np.append(day,a6)
        hr=np.append(hr,a7)
        mn=np.append(mn,a8)
        sc=np.append(sc,a9)
        
    data_jpg=np.column_stack((yr, mnt, day, hr, mn, sc, lon, lat, alt))
    
    #Select reference point
    ref=data_jpg[int(data_jpg.shape[0]/2)].copy()
    year, month, day, hour, minute, second=ref[:6]
    lon, lat = query_specific_time(path_gps_database, year, month, day, hour, minute, second)
    
    ref[6]=lon
    ref[7]=lat
    
    #Correct the longitude of the eastern and western hemispheres
    if (ref[6]<0)&(data_jpg[0,6]>0):
        data_jpg[:,6]=-data_jpg[:,6]
    
    #Perform drift correction as follows
    #The above reference positions, Position of image processing, Ship position corresponding to the time of image processing
    
    #Data_jpg_cor is the corrected image information
    data_jpg_cor=data_jpg.copy()
    
    for i in tqdm(range(data_jpg.shape[0])):
        
        ref_lat=ref[7]
        ref_lon=ref[6]
        
        loc_img_lat=data_jpg[i,7].item()
        loc_img_lon=data_jpg[i,6].item()
        
        year, month, day, hour, minute, second=data_jpg[i,:6]
        loc_gps_lon, loc_gps_lat = query_specific_time(path_gps_database, year, month, day, hour, minute, second)
        
        loc_img_lat_cor, loc_img_lon_cor=DriftCorr(ref_lat, ref_lon, loc_gps_lat, loc_gps_lon, loc_img_lat, loc_img_lon)
        
        data_jpg_cor[i,6]=loc_img_lon_cor
        data_jpg_cor[i,7]=loc_img_lat_cor
    
    #Save the corrected image information to csv: filename, lat, lon, alt
    corrected_data = np.column_stack((file_names, data_jpg_cor[:, 7], data_jpg_cor[:, 6], data_jpg_cor[:, 8]))
    np.savetxt('corrected_image_info.csv', corrected_data, delimiter=',', fmt='%s', header='filename,lat,lon,alt', comments='')

    

if __name__ == "__main__":
    main()
