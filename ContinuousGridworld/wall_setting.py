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
    return wall

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
    return wall