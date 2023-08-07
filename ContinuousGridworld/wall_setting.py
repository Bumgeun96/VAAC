import random

def map_1():
    wall = []
    for i in range(101):
        wall.append((50,i))
        wall.append((49,i))
        wall.append((51,i))
        wall.append((i,50))
        wall.append((i,49))
        wall.append((i,51))
    wall = list(set(wall))
    door = []
    for i in [x for x in range(20, 30)] + [x for x in range(71, 81)]:
        door.append((50,i))
        door.append((49,i))
        door.append((51,i))
        door.append((i,50))
        door.append((i,49))
        door.append((i,51))
    for d in door:
        wall.remove(d)
    points = wall_line_detect(wall)
    line_points = making_lines(points)
    return wall, line_points

def map_2():
    wall = []
    for i in range(101):
        wall.append((23,i))
        wall.append((24,i))
        wall.append((25,i))
        wall.append((i,23))
        wall.append((i,24))
        wall.append((i,25))
        
        wall.append((49,i))
        wall.append((50,i))
        wall.append((51,i))
        wall.append((i,49))
        wall.append((i,50))
        wall.append((i,51))
        
        wall.append((75,i))
        wall.append((76,i))
        wall.append((77,i))
        wall.append((i,75))
        wall.append((i,76))
        wall.append((i,77))
    wall = list(set(wall))
    door = []
    for i in [x for x in range(9, 14)]\
        + [x for x in range(35, 40)]\
            + [x for x in range(61, 66)]\
                + [x for x in range(86, 91)]:
        door.append((23,i))
        door.append((24,i))
        door.append((25,i))
        door.append((i,23))
        door.append((i,24))
        door.append((i,25))
        
        door.append((49,i))
        door.append((50,i))
        door.append((51,i))
        door.append((i,49))
        door.append((i,50))
        door.append((i,51))
        
        door.append((75,i))
        door.append((76,i))
        door.append((77,i))
        door.append((i,75))
        door.append((i,76))
        door.append((i,77))
    for d in door:
        wall.remove(d)
    points = wall_line_detect(wall)
    line_points = making_lines(points)
    return wall, line_points

def wall_line_detect(walls):
    wall_edges = []
    for wall in walls:
        x,y =  wall
        p1 = (x+0.5,y+0.5)
        p2 = (x-0.5,y+0.5)
        p3 = (x-0.5,y-0.5)
        p4 = (x+0.5,y-0.5)
        wall_edges.append([p1,p2,p3,p4])
    points = []
    while len(wall_edges) != 0:
        parts = []
        edge = random.choice(wall_edges)
        parts = [edge]
        wall_edges.remove(edge)
        for part in parts:
            for item in part:
                for edge in wall_edges:
                    if item in edge:
                        parts.append(edge)
                        wall_edges.remove(edge)
        result_list = [item for sublist in parts for item in sublist]
        point =[]
        for element in result_list:
            if result_list.count(element) % 2 == 0:
                pass
            else:
                point.append(element)
        points.append(list(set(point)))
    return points

def making_lines(boundary_points):
    line_points = []
    for boundary_point in boundary_points:
        boundary_point= [list(i) for i in boundary_point]
        boundary_point = sorted(boundary_point, key=lambda x: (x[1], x[0]))
        result = [boundary_point[i:i+2] for i in range(0, len(boundary_point), 2)]
        line_points += result
        boundary_point = sorted(boundary_point, key=lambda x: (x[0], x[1]))
        result = [boundary_point[i:i+2] for i in range(0, len(boundary_point), 2)]
        line_points += result
    return line_points