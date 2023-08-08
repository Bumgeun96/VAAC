import math

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

def checking_physics(agent_location,previous_agent_location,boundary_points):
    end_point = agent_location
    for p in boundary_points:
        end_point = is_cross_line(previous_agent_location,end_point,p[0],p[1])
    return end_point
