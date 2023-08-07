import math

def find_near_wall(agent_location,walls):
    temp = []
    for wall in walls:
        d = math.sqrt((agent_location[0] - wall[0])**2+(agent_location[1] - wall[1])**2)
        if d < 2.2:
            temp.append(wall)
    return temp


def is_cross_line(p1,p2,p3,p4):
    # p1 = start point of the agent
    # p2 = end point of the agent
    x = 0
    y = 1
    
    if p3[x] == p4[x] and p1[x] != p2[x]:
        x1 = abs(p3[x]-p1[x])
        x2 = abs(p3[x]-p2[x])
        x3 = abs(p1[x]-p2[x])
        if x1+x2 == x3:
            xx = p3[x]
            yy = p1[y]+(p2[y]-p1[y])*(x1/x3)
            if min(p3[y],p4[y])<=yy and max(p3[y],p4[y])>=yy:
                return [xx,yy]
            else:
                return p2
        else:
            return p2
    elif p3[y] == p4[y] and p1[y] != p2[y]:
        y1 = abs(p3[y]-p1[y])
        y2 = abs(p3[y]-p2[y])
        y3 = abs(p1[y]-p2[y])
        if y1+y2 == y3:
            yy = p3[y]
            xx = p1[x]+(p2[x]-p1[x])*(y1/y3)
            if min(p3[x],p4[x])<=xx and max(p3[x],p4[x])>=xx:
                return [xx,yy]
            else:
                return p2
        else:
            return p2
    else:
        return p2
        
def is_in_square(center,start_point,end_point):
    x = center[0]
    y = center[1]
    p1 = [x+0.5,y+0.5]
    p2 = [x-0.5,y+0.5]
    p3 = [x-0.5,y-0.5]
    p4 = [x+0.5,y-0.5]
    if end_point[0] <= x+0.5 \
        and end_point[0] >= x-0.5 \
            and end_point[1] <= y+0.5 \
                and end_point[1] >= y-0.5:
                    for p in [[p1,p2],[p2,p3],[p3,p4],[p4,p1]]:
                        end_point = is_cross_line(start_point,end_point,p[0],p[1])
                    return end_point
    else:
        return end_point
    
# def checking_physics(agent_location,previous_agent_location,wall):
#     walls = find_near_wall(agent_location,wall)
#     if len(walls) == 0:
#         return agent_location
#     for wall in walls:
#         agent_location = is_in_square([wall[0],wall[1]],previous_agent_location,agent_location)
#     for wall in walls:
#         agent_location = is_in_square([wall[0],wall[1]],previous_agent_location,agent_location)
#     return agent_location

def checking_physics(agent_location,previous_agent_location,boundary_points):
    end_point = agent_location
    for p in boundary_points:
        end_point = is_cross_line(previous_agent_location,end_point,p[0],p[1])
    return end_point
