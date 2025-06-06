from bs4 import BeautifulSoup #pip install bs4
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import pydicom as pydicom
import scipy.ndimage
import xml.etree.ElementTree as ET
import re
import lxml #pip install lxml /or python -m pip install lxml /or pip3 install lxml
from scipy.interpolate import RegularGridInterpolator, interpn
from scipy.interpolate import CubicSpline
from scipy import interpolate
from scipy import ndimage
import cv2
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom
import copy
from tqdm import tqdm

def read_dicom(dcm_path): #reads isotropic
    files = os.listdir(dcm_path)
    files = [f for f in files if f.endswith('.dcm')]
    files.sort()

    pixel_data_list = []
    for dcm_file in files:
        dcm = pydicom.dcmread(os.path.join(dcm_path, dcm_file))
        pixel_data_list.append(dcm.pixel_array)
    volume = np.stack(pixel_data_list)
    # print(volume.shape)
    volume = volume.transpose(2, 1, 0)
    # print(volume.shape)
    # volume = np.array(pixel_data_list)

    voxel_spacing = np.array([dcm.PixelSpacing[0], dcm.PixelSpacing[1], 0.5])
    # print(voxel_spacing)
    min_spacing = min(voxel_spacing)
    target_spacing = [min_spacing]*3

    resampling_factor = voxel_spacing / target_spacing
    # print(resampling_factor)
    isotropic_volume = zoom(volume, resampling_factor, order=1)
    # print(isotropic_volume.shape)

    data_array = np.flip(isotropic_volume, axis=2) #add by shjung. flip the order of slices
    
    return data_array,voxel_spacing,resampling_factor

def get_start_end_ID(mxml_path):
    search_str = "mRCA"
    tree = ET.parse(mxml_path)
    root = tree.getroot()
    layer_elem = None
    for elem in root.iter("LAYER"):
        for info_elem in elem.findall("./INFO[@id='Name']"):
            if info_elem.attrib["value"] == search_str:
                layer_elem = elem
                break
        if layer_elem is not None:
            break

    # if the search string is found, get the parent element
    if layer_elem is not None:
        layer_id = layer_elem.attrib["id"]
        # print("LAYER id of {}: {}".format(search_str, layer_id))
    else:
        print("{} not found in data model XML file.".format(search_str))

    layer_id_to_search='"'+layer_id+'"'

    for pos in root.findall("./OBJECT/LAYER/[@id={}]/INFO/[@id='EndPos']".format(layer_id_to_search)):
        endPos=float(pos.attrib['value'])

    for pos in root.findall("./OBJECT/LAYER/[@id={}]/INFO/[@id='StartPos']".format(layer_id_to_search)):
        startPos= float(pos.attrib['value'])
    #convert these pos points to ids in xml file
    startID=int(round(startPos*2,0))
    endID=int(round(endPos*2,0))
    return startID, endID

# import re
def get_points(mxml_path, xml_path, sp):
    startID, endID= get_start_end_ID(mxml_path)
    with open(xml_path, 'r') as f:
        data = f.read()
    soup = BeautifulSoup(data, "xml")
    temp=[]
    for tag in soup.find_all(re.compile("id")):
        # print(tag.string)
        for sibling in tag.parent.next_siblings:
            temp.append(str(sibling.string))
            # print(str(sibling.string))
        # print(tag.parent.next_siblings)

    for i in range(len(temp)-1, -1, -1):
        if temp[i] == '\n':
            del temp[i]

    points=[]
    for i in range(startID, endID+1):
        if i < len(temp):
            n=temp[i].split(' ')
            points.append([n[0],n[1],n[2]])
    points=np.array(points).astype('float')
    for pt in points:
        pt[0]=int(pt[0]/sp[0])
        pt[1]=int(pt[1]/sp[0])
        pt[2]=int(pt[2]/sp[0])
        # pt[2]=int(pt[2]/0.5)
    return points

def get_sampled(points, samples):
    distance=get_distance(points)
    centerline=sample_by_distance(points, distance, samples)
    # len(centerline)
    return centerline

def get_distance(arr):
    dist = 0
    for i in range(1, len(arr)):
        dist += np.linalg.norm(arr[i] - arr[i-1])
        # print(i, i-1)
    return dist

def sample_by_distance(arr, distance ,samples):
    pp = []  # sampled points
    dist=distance/samples
    ref_point = arr[0]  # reference point
    pp.append(ref_point)
    for i in range(1, len(arr)-1):
        d = np.linalg.norm(arr[i] - ref_point)  # distance from reference point to current point
        if np.isclose(d, dist, rtol=0.2):  # if the distance is close to the desired distance
            pp.append(arr[i])  # add current point to sampled points
            ref_point = arr[i]  # update reference point to current point
    if len(pp)<samples:
        pp.append(arr[-1])
    if len(pp)>samples:
        n=len(pp)-samples
        pp=pp[:-n]
    return np.array(pp)

PI = np.pi
def get_theta_phi_from_normal(normal):
    x, y, z = normal
    xy=np.sqrt(x**2+y**2)
    theta = -(np.arctan2(xy,z))
    phi = -(np.arctan2(x,y))
#     if z<0:
#         theta = (np.arctan2(xy,z))-PI
#     elif z==0:
#         print('here')
#     else:
#         theta = (np.arctan2(xy,z))
#     if x>0:
#         phi = (np.arctan2(x,y))
#     else:
#         if y>0 or y==0:
#             phi = (np.arctan2(x,y))-PI
#         else:
#             phi =(np.arctan2(x,y))+PI
    
#     print(theta, phi)

    return theta, phi
PI = np.pi
def rotate(orgVector, axis, theta):
    th = theta * PI/180
    # th=theta

    if axis == 1: # z-axis
        matrix = np.array([[np.cos(th), -np.sin(th), 0],
                           [np.sin(th), np.cos(th), 0],
                           [0, 0, 1]])
    elif axis == 2: # y-axis
        matrix = np.array([[np.cos(th), 0, np.sin(th)],
                           [0, 1, 0],
                           [-np.sin(th), 0, np.cos(th)]])
    else: # x-axis
        matrix = np.array([[1, 0, 0],
                           [0, np.cos(th), -np.sin(th)],
                           [0, np.sin(th), np.cos(th)]])

    roatatedVec = matrix.dot(orgVector.T).T

    return roatatedVec

def window_image(image, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    
    return window_image

# def get_normalized(scann,mn,mx):
#     scan = copy.copy(scann)
#     mn = max(mn,np.amin(scan))
#     mx = min(mx,np.amax(scan))
#     np.clip(scan, mn, mx, out=scan)
#     d = mx - mn
#     scan = (scan-mn)/d
#     return scan


def generate_patches(centerline, scan,size):
    patch_size = size
    patch_half = patch_size // 2
    points2=[]
    delta=[]
    corners=[]
    normals=[]
    for index in range(len(centerline)):

        if index == len(centerline) - 1:
            normal = centerline[index] - centerline[index-1]
        else:
            normal = centerline[index+1] - centerline[index]
        # print(normal)
        normals.append(normal)

        deltaTheta, deltaPhi = get_theta_phi_from_normal(normal)

        x, y = np.meshgrid(np.arange(-patch_half, patch_half), np.arange(-patch_half, patch_half))
        z = np.zeros_like(x)
        # print(x.shape)

        rotated_vectors = np.array([x.flatten(), y.flatten(), z.flatten()]).T
        rotated_vectors = rotate(rotated_vectors, 3, (deltaTheta * 180 / PI))
        rotated_vectors = rotate(rotated_vectors, 1, (deltaPhi * 180 / PI))
        # print(rotated_vectors.shape)

        u = np.round(rotated_vectors[:, 0] + centerline[index][0], decimals=2)#.astype(int)
        v = np.round(rotated_vectors[:, 1] + centerline[index][1], decimals=2)#.astype(int)
        w = np.round(rotated_vectors[:, 2] + centerline[index][2], decimals=2)#.astype(int)
        
        #reshape to get corner coords
        u_rs = u.reshape(x.shape)
        v_rs = v.reshape(x.shape)
        w_rs = w.reshape(x.shape)
        # print(u_rs)
        # print(u_rs[0,0], u_rs[0, 63], u_rs[63, 0], u_rs[63,63])
        # c1 = [u_rs[0,0], v_rs[0,0], w_rs[0,0]]
        # c2 = [u_rs[0, 63], v_rs[0, 63], w_rs[0, 63]]
        # c3 = [u_rs[63, 0], v_rs[63, 0], w_rs[63, 0]]
        # c4 = [u_rs[63,63],  v_rs[63,63],  w_rs[63,63]]

        u1 = u_rs[0,0], u_rs[0, size-1], u_rs[size-1, 0], u_rs[size-1,size-1]
        v1 = v_rs[0,0], v_rs[0, size-1], v_rs[size-1, 0], v_rs[size-1,size-1]
        w1 = w_rs[0,0], w_rs[0, size-1], w_rs[size-1, 0], w_rs[size-1,size-1]
        corners.append(np.array([u1, v1, w1]).T)

        delta.append([deltaTheta, deltaPhi])
        points2.append(np.array([u,v, w]).T )#access the volume by order of (y, x, z)
    

    patches=[]
    patch_shape = (size, size)

    for i in range(len(points2)):
        output_patch=ndimage.map_coordinates(scan, points2[i].T, order=1, mode='nearest')
        patches.append(np.reshape(output_patch, patch_shape))
    return np.array(patches), points2, delta, corners, normals

def compute_distances(A, B, C):
    temp = np.linalg.norm(C - B, axis=1)
    d = (C - B) / temp[:, np.newaxis]
    v = A - B
    t = np.einsum('ij,ij->i', v, d)
    P = B + d * t[:, np.newaxis]
    return np.linalg.norm(A - P, axis=1)

def bbox(corner1, corner2):
    conc = np.concatenate(((np.round(corner1)).astype(int), (np.round(corner2)).astype(int)), axis=0)
    min_values = np.min(conc, axis=0)
    max_values = np.max(conc, axis=0)
    # print('bbox')

    return min_values, max_values

def add_noise(img):
    noise_factor = 240  
    noisy_img = np.clip(img + np.random.normal(0, noise_factor, size=img.shape), 0, 65535).astype(np.uint16)
    return noisy_img

def inverse_things(lineStart, lineEnd, deltaTheta, deltaPhi, centerline):
    lineStart-= centerline
    lineEnd-= centerline
    
    inv_rotated_lineStart = rotate(lineStart, 1, (-deltaPhi * 180 / PI))
    inv_rotated_lineStart = rotate(inv_rotated_lineStart, 3, (-deltaTheta * 180 / PI))
    
    # print(inv_rotated_lineStart.shape)
    inv_rotated_lineStart = inv_rotated_lineStart[:, :2]+32
    # print(inv_rotated_lineStart)

    inv_rotated_lineEnd = rotate(lineEnd, 1, (-deltaPhi * 180 / PI))
    inv_rotated_lineEnd = rotate(inv_rotated_lineEnd, 3, (-deltaTheta * 180 / PI))
    
    inv_rotated_lineEnd = inv_rotated_lineEnd[:, :2]+32

    return inv_rotated_lineStart, inv_rotated_lineEnd

#final

def insert_patch(scan, norm_patches, points, corners, normals, centerline):
    sc=scan.copy()
    # factor=0.8
    for i in range(len(norm_patches)):
        patch=(norm_patches[i]).flatten()
        p = points[i].reshape((-1, 3))
        # print(len(patch), len(p))
        # patch=org[i].flatten()
        for j in range(len(p)):
            x,y,z = p[j].astype(int)
            sc[x,y,z] = patch[j]
            
    rein=sc.copy()
    n=len(norm_patches)

    for index in tqdm(range(1, n)):
        points1= points[index-1]
        points2= points[index]
        # patch1= patches[index-1]
        # patch2= patches[index]
        min_values, max_values = bbox(corners[index-1], corners[index])

        coordinates = np.array(np.meshgrid(range(min_values[0], max_values[0]),
                                            range(min_values[1], max_values[1]),
                                            range(min_values[2], max_values[2]))).T.reshape(-1, 3)

        lineStart = []
        lineEnd = []
        xyz = []
        lineStart_xy = []
        lineEnd_xy = []
        dist = []
        shortest_indexes = []

        for coord in coordinates:
                x,y,z=coord[0], coord[1], coord[2]
                distances = compute_distances(np.array([x, y, z]), points1 ,points2)
                shortest_index = np.argmin(distances)
                s = distances[shortest_index]
                if s<0.2:
                    lineStart.append(points1[shortest_index])
                    lineEnd.append(points2[shortest_index])
                    xyz.append(coord)

        lineStart = np.array(lineStart)
        lineEnd = np.array(lineEnd)
        xyz = np.array(xyz)
        # normal=normals[index]
        # deltaTheta, deltaPhi = get_theta_phi_from_normal(normal)
        # lineStart_xy, lineEnd_xy = inverse_things(lineStart, lineEnd, (delta[index][0]), (delta[index][1]), centerline[index])

        length1 = np.linalg.norm(xyz - lineStart, axis=1)
        length2 = np.linalg.norm(xyz - lineEnd, axis=1)
        weight1 = 1 - (length1 / (length1 + length2))
        weight2 = 1 - (length2 / (length1 + length2))

        vox1 = ndimage.map_coordinates(sc, lineStart.T , order=3, mode='mirror')
        vox2 = ndimage.map_coordinates(sc, lineEnd.T , order=3, mode='mirror')
        aver_val = ((vox1 * weight1) + (vox2 * weight2) + 0.5).astype(int)
        
        for i in range(len(xyz)):
            x,y,z = xyz[i]
            rein[x,y,z]= aver_val[i]

    smoothed_array = gaussian_filter(rein, sigma=2)
    image =smoothed_array.copy()
    image2=scan.copy()
    sharpened = cv2.addWeighted(image2, 0.7, image, 1.2, 0)

    return sharpened

# path = "/home/amal/Desktop/motion_artifact/reinsert/shin_hyun_heui/50_reinsert/"
# sharpened=sharpened.astype('uint16')
# save_dicom(path, dcm_path, sharpened, resampling_factor)
    
# def insert_patch(scan, patches, points, corners, normals, centerline):
    # sc=scan.copy()
    # max_val= np.max(scan)
    # n=len(patches)
    # # index=1
    # factor=0.6
    # for index in tqdm(range(4, 9)):
    #     points1= points[index-1]
    #     points2= points[index]
    #     patch1= patches[index-1]
    #     patch2= patches[index]
    #     min_values, max_values = bbox(corners[index-1], corners[index])

    #     coordinates = np.array(np.meshgrid(range(min_values[0], max_values[0]),
    #                                         range(min_values[1], max_values[1]),
    #                                         range(min_values[2], max_values[2]))).T.reshape(-1, 3)

    #     lineStart = []
    #     lineEnd = []
    #     xyz = []
    #     lineStart_xy = []
    #     lineEnd_xy = []
    #     for coord in coordinates:
    #         x,y,z=coord[0], coord[1], coord[2]
    #         distances = compute_distances(np.array([x, y, z]), points1, points2)
    #         shortest_index = np.argmin(distances)
    #         s = distances[shortest_index]
    #         if s<2:
    #             lineStart.append(points1[shortest_index])
    #             lineEnd.append(points2[shortest_index])
    #             xyz.append(coord)

    #     lineStart = np.array(lineStart)
    #     lineEnd = np.array(lineEnd)
    #     xyz = np.array(xyz)
    #     normal=normals[index]
    #     deltaTheta, deltaPhi = get_theta_phi_from_normal(normal)
    #     lineStart_xy, lineEnd_xy = inverse_things(lineStart, lineEnd, deltaTheta, deltaPhi, centerline[index])

    #     length1 = np.linalg.norm(xyz - lineStart, axis=1)
    #     length2 = np.linalg.norm(xyz - lineEnd, axis=1)
    #     weight1 = 1 - (length1 / (length1 + length2))
    #     weight2 = 1 - (length2 / (length1 + length2))
    #     patch1 = (patch1 / 255.0) * max_val
    #     patch2 = (patch2 / 255.0) * max_val

    #     patch1 = patch1 * factor
    #     patch2 = patch2 * factor
    #     patch1 = np.clip(patch1, 0, 255)
    #     patch2 = np.clip(patch2, 0, 255)

    #     vox1 = ndimage.map_coordinates(patch1, lineStart_xy.T , order=3, mode='mirror', prefilter=True)
    #     vox2 = ndimage.map_coordinates(patch2, lineEnd_xy.T , order=3, mode='mirror', prefilter=True)
    #     aver_val = ((vox1 * weight1) + (vox2 * weight2) + 0.5).astype(int)
    #     # print(aver_val)
        
    #     for i in range(len(xyz)):
    #         x,y,z = xyz[i]
    #         sc[x,y,z]= aver_val[i]
    #     # sc[xyz[:, 0], xyz[:, 1], xyz[:, 2]] = aver_val

    # return sc

def convert_dicom_to_0_255(dicom_image):
    # Define the desired range
    new_min = 0
    new_max = 255

    # Calculate the current range of pixel values in the DICOM image
    current_min = np.min(dicom_image)
    current_max = np.max(dicom_image)

    # Perform linear transformation to map pixel values to the new range
    converted_image = (dicom_image - current_min) / (current_max - current_min) * (new_max - new_min) + new_min

    # Clip values to ensure they are in the valid range [0, 255]
    converted_image = np.clip(converted_image, 0, 255)

    return converted_image.astype(np.uint8)

import numpy as np

def convert_0_255_to_dicom(converted_image, original_min, original_max):
    # Define the original range of pixel values in the DICOM image
    new_min = 0
    new_max = 255

    # Perform inverse linear transformation to map pixel values back to the original range
    dicom_image = (converted_image - new_min) / (new_max - new_min) * (original_max - original_min) + original_min

    # Clip values to ensure they are in the original range
    dicom_image = np.clip(dicom_image, original_min, original_max)

    return dicom_image


def save_dicom(save_path, dcm_path, new_scan, resampling_factor):
    files = os.listdir(dcm_path)
    files = [f for f in files if f.endswith('.dcm')]
    files.sort()

    volume = zoom(new_scan, 1.0/resampling_factor, order=1)
    volume = np.flip(volume, axis=2)
    volume = volume.transpose(2, 1, 0)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(volume.shape[0]):
        dcm = pydicom.dcmread(os.path.join(dcm_path, files[i]))
        ds = dcm.copy()
        img1 = volume[i,:,:].copy()
        # img1 = (volume[:, :, i] * np.iinfo(np.uint16).max).astype(np.uint16)
        ds.SmallestImagePixelValue = np.min(img1)
        ds.LargestImagePixelValue = np.max(img1)
        # ds.WindowCenter = -900
        # ds.WindowWidth = 290
        ds.PixelData = img1.tobytes()
        filename = save_path+str(i)+".dcm"
        ds.save_as(filename)

def main():
    
    phases = ['0', '10', '20', '40', '50', '60', '70', '80']
    n = 32
    for phase in tqdm(phases, desc='Processing phases', unit='phase'):

        base_path = "/home/amal/Desktop/motion_artifact/cyclegan_results/images/4DCT_RCA_DB_3_P100_SUNG SI JE_"+ phase +'_'
        xml_path= "/home/amal/Desktop/motion_artifact/dataset2/4DCT_RCA_DB_3_P100/SUNG SI JE/ANALYSIS/" + phase + "/Analysis" + phase +"/pathline.xml"
        data_xml= "/home/amal/Desktop/motion_artifact/dataset2/4DCT_RCA_DB_3_P100/SUNG SI JE/ANALYSIS/" + phase + "/Analysis" + phase + "/data_model.xml"
        dcm_path= "/home/amal/Desktop/motion_artifact/dataset2/4DCT_RCA_DB_3_P100/SUNG SI JE/DICOM/" + phase + "/"
        path = "/home/amal/Desktop/motion_artifact/reinsert/SUNG SI JE/" + phase + "/"
        
        scan, sp, resampling_factor = read_dicom(dcm_path) 

        points = get_points(data_xml, xml_path, sp) #mid rca points from xml file using data model file
        spline_interp = [CubicSpline(np.arange(len(points)), points[:, i]) for i in range(3)]
        interpolated_arr = np.array([spline_interp[i](np.linspace(0, len(points), len(points)*10)) for i in range(3)]).T
        centerline = get_sampled(interpolated_arr, 20)
        original_min, original_max = np.min(scan), np.max(scan)
        scan = convert_dicom_to_0_255(scan)
        _, points, delta, corners, normals = generate_patches(centerline, scan, 32)

        norm_patches = []
        for i in range(0, 20):  
            filename = f"{base_path}{i}_fake.png"
            img = cv2.imread(filename)
            img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (64, 64))
            start_x = (img.shape[1] - n) // 2
            start_y = (img.shape[0] - n) // 2
            cropped_img = img[start_y:start_y + n, start_x:start_x + n]
            norm_patches.append(cropped_img)
        norm_patches = np.array(norm_patches)

        sharpened = insert_patch(scan, norm_patches, points, corners, normals, centerline)
        sharpened = sharpened.astype('uint16')
        save_dicom(path, dcm_path, sharpened, resampling_factor)
    

if __name__ == "__main__":
    main()