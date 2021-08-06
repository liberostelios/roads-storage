import sys
import threading
import time
import concurrent.futures as cf
import json
import math
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import pprint
pp = pprint.PrettyPrinter(indent=2)

filename = "/Users/ken/Downloads/montreal_final_merged_intersections.json"
output_path = "/Users/ken/Downloads/road_stats.csv"
search_radius = 20.0 # for each road, use buildings roughly this distance away (bounding boxes)
view_height = 1.7 # how high the viewpoint is
slice_interval = 5.0 # distance between slices
angles_to_test = 36 # how many angles to test

print_lock = threading.Lock()

def process_road(road_id):
  global roads_processed
  global roads
  global buildings
  with print_lock:
    print("Thread " + str(threading.get_ident()) + ": processing road '" + str(road_id) + "' (" + str(roads_processed) + "/" + str(len(roads)) + ")...")
    roads_processed += 1

  # For each road, find nearby buildings
  nearby_buildings = []
  for building_id in list(buildings):
    if bbox_distance(roads[road_id]["surface"], buildings[building_id]["surface"]) < search_radius:
      nearby_buildings.append(building_id)
  if len(nearby_buildings) == 0:
    return

  # Create one mesh with nearby buildings
  vertices = []
  faces = []
  num_triangles = 0
  for building_id in roads[road_id]["nearby_buildings"]:
    for i in range(int(len(buildings[building_id]["surface"].faces)/4)):
      vertices.extend([buildings[building_id]["surface"].points[buildings[building_id]["surface"].faces[4*i+1]],
                       buildings[building_id]["surface"].points[buildings[building_id]["surface"].faces[4*i+2]],
                       buildings[building_id]["surface"].points[buildings[building_id]["surface"].faces[4*i+3]]])
      faces.extend([3, 3*num_triangles, 3*num_triangles+1, 3*num_triangles+2])
      num_triangles += 1
  nearby_buildings = pv.PolyData(vertices, faces, num_triangles)

def bbox_distance(pd1, pd2):
    xmin1 = pd1.bounds[0]
    xmax1 = pd1.bounds[1]
    ymin1 = pd1.bounds[2]
    ymax1 = pd1.bounds[3]
    zmin1 = pd1.bounds[4]
    zmax1 = pd1.bounds[5]
    xmin2 = pd2.bounds[0]
    xmax2 = pd2.bounds[1]
    ymin2 = pd2.bounds[2]
    ymax2 = pd2.bounds[3]
    zmin2 = pd2.bounds[4]
    zmax2 = pd2.bounds[5]
    if xmin1 > xmax2:
        xdist = xmin1-xmax2
    elif xmax1 < xmin2:
        xdist = xmin2-xmax1
    else:
        xdist = 0.0
    if ymin1 > ymax2:
        ydist = ymin1-ymax2
    elif ymax1 < ymin2:
        ydist = ymin2-ymax1
    else:
        ydist = 0.0
    if zmin1 > zmax2:
        zdist = zmin1-zmax2
    elif zmax1 < zmin2:
        zdist = zmin2-zmax1
    else:
        zdist = 0.0
    return math.sqrt(xdist*xdist + ydist*ydist + zdist*zdist)

def main():

  # Read data
  with open(filename) as file:
    cm = json.load(file)
  print("Read " + str(len(cm["CityObjects"])) + " city objects")

  # Totals
  totals = {}
  road_totals = {}
  for co_id, co in cm["CityObjects"].items():
      if co["type"] in totals:
          totals[co["type"]] += 1
      else:
          totals[co["type"]] = 1
      if co["type"] == "Road":
          if co["attributes"]["road_type"] in road_totals:
              road_totals[co["attributes"]["road_type"]] += 1
          else:
              road_totals[co["attributes"]["road_type"]] = 1
        
  print("Totals:")
  for co_type, co_total in totals.items():
      print("\t" + str(co_type) + ": " + str(co_total))
  print("Road totals:")
  for road_type, type_total in road_totals.items():
      print("\t" + str(road_type) + ": " + str(type_total))

  # Load roads and buildings into PyVista
  global roads
  roads = {}
  global buildings
  buildings = {}
  for co_id in list(cm["CityObjects"]):
      co = cm["CityObjects"][co_id]
      
      if co["type"] == "Road":
          if co["attributes"]["road_type"] == "Main road":
              roads[co_id] = {}
              for geom in co["geometry"]:
                  if geom["type"] == "MultiLineString":
                      geom_line = geom
                  if geom["type"] == "MultiSurface":
                      geom_surface = geom

              lines = []
              lines_vertices = []
              num_lines = 0
              for segment in geom_line["boundaries"]:
                  lines.extend([2, 2*num_lines, 2*num_lines+1])
                  lines_vertices.extend([cm["vertices"][segment[0]], cm["vertices"][segment[1]]])
                  num_lines += 1
              if len(lines) > 0:
                  mesh_lines = pv.PolyData(lines_vertices, lines=lines, n_lines=len(geom_line["boundaries"]))
                  roads[co_id]["line"] = mesh_lines

              surface = []
              surface_vertices = []
              num_triangles = 0
              for triangle in geom_surface["boundaries"]:
                  surface.extend([3, 3*num_triangles, 3*num_triangles+1, 3*num_triangles+2])
                  surface_vertices.extend([cm["vertices"][triangle[0][0]], cm["vertices"][triangle[0][1]], cm["vertices"][triangle[0][2]]])
                  num_triangles += 1
              if len(surface) > 0:
                  mesh_surface = pv.PolyData(surface_vertices, surface, len(geom_surface["boundaries"]))
                  roads[co_id]["surface"] = mesh_surface
              
      if co["type"] == "Building":
          buildings[co_id] = {}
          for geom in co["geometry"]:
              if geom["type"] == "Solid":
                  geom_solid = geom
          
          surface = []
          surface_vertices = []
          num_triangles = 0
          for triangle in geom_solid["boundaries"][0]:
              surface.extend([3, 3*num_triangles, 3*num_triangles+1, 3*num_triangles+2])
              surface_vertices.extend([cm["vertices"][triangle[0][0]], cm["vertices"][triangle[0][1]], cm["vertices"][triangle[0][2]]])
              num_triangles += 1
          if len(surface) > 0:
              mesh_surface = pv.PolyData(surface_vertices, surface, len(geom_surface["boundaries"]))
              buildings[co_id]["surface"] = mesh_surface
              
  del cm

  # Remove roads and buildings without required geometries
  for road_id in list(roads):
      if "surface" not in roads[road_id] or "line" not in roads[road_id]:
          roads.pop(road_id)
          
  for building_id in list(buildings):
      if "surface" not in buildings[building_id]:
          buildings.pop(building_id)
        
  print("Elements that can be used for processing:")
  print("\t" + str(len(roads)) + " roads")
  print("\t" + str(len(buildings)) + " buildings")

  # Compute widths
  viewpoint = np.array([0, 0, view_height])
  angle_interval = int(360/angles_to_test)
  output_file = open(output_path, "w")
  output_file.write("road_id,")
  output_file.write("slices,")
  output_file.write("nearby_buildings,")
  output_file.write("horizontal_distance_left_mean,horizontal_distance_left_median,horizontal_distance_left_min,")
  output_file.write("horizontal_distance_right_mean,horizontal_distance_right_median,horizontal_distance_right_min,")
  output_file.write("minimum_diagonal_distance_mean,minimum_diagonal_distance_median,minimum_diagonal_distance_min,")
  output_file.write("maximum_diagonal_distance_mean,maximum_diagonal_distance_median,maximum_diagonal_distance_max,")
  output_file.write("maximum_obscured_angle_mean,maximum_obscured_angle_median,maximum_obscured_angle_max,")
  output_file.write("sky_visibility_mean,sky_visibility_median,sky_visibility_min,sky_visibility_max\n")

  global roads_processed
  roads_processed = 1
  with cf.ThreadPoolExecutor(max_workers=8) as executor:
    executor.map(process_road, list(roads)[0:101])
  print("Done!")

if __name__ == '__main__':
    main()