import numpy as np

def toTriangularIncides(indices):
    triangles = []

    for square in indices:
        triangles.append([square[0], square[1], square[2]])
        triangles.append([square[1], square[3], square[2]])
    return triangles

def generateCylinder(heigth=1.0, radius = 1.0, faces = 4):
    x = 0
    y = 0 
    PI = np.pi
    step = 2*PI / faces
    angle = 0

    vertices = []
    indices = []
    count = 0
    index_aux = 0

    # genera costado del cilindro
    while(angle < 2*PI):
        x = np.cos(angle)
        y = np.sin(angle)

        vertices.append([x,y,heigth])
        vertices.append([x,y,0.0])

        angle += step
        count += 2
        index_aux += 2

        if(count % 4 == 0):
            count = 2
            indices.append([index_aux-4, index_aux-3, index_aux-2, index_aux-1])

    indices.append([index_aux-2, index_aux-1, 0, 1])

    # genera tapas
    # tapa de arriba
    angle = 0
    count = 0
    
    index_tapa_1 = index_aux
    vertices.append([0,0,heigth])

    indices = toTriangularIncides(indices)

    while(angle < 2*PI):
        x = np.cos(angle) * radius
        y = np.sin(angle) * radius

        vertices.append([x,y,heigth])
        angle += step
        count += 1
        index_aux += 1

        if(count %2 == 0):
            count = 1
            indices.append([index_tapa_1, index_aux-1, index_aux])

    indices.append([index_tapa_1, index_aux, index_tapa_1+1])

    # tapa de abajo
    angle = 0
    count = 0

    index_aux += 1
    index_tapa_2 = index_aux
    vertices.append([0, 0, 0])

    while(angle < 2*PI):
        x = np.cos(angle) * radius
        y = np.sin(angle) * radius

        vertices.append([x, y, 0])
        angle += step
        count += 1
        index_aux += 1

        if(count % 2 == 0):
            count = 1
            indices.append([index_tapa_2, index_aux-1, index_aux])

    indices.append([index_tapa_2, index_aux, index_tapa_2+1])

    # print(np.asarray(vertices))
    # print(np.asarray(indices))

    vertices = np.asarray(vertices)
    indices = np.asarray(indices)

    indices = indices.astype(np.uint32)
    vertices = vertices.astype(np.float32)

    return ([np.round(vertices, decimals=2), np.asarray(indices)])

# generateCylinder()
