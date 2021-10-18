# coding=utf-8
"""Tarea 3"""

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import sys
import os.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import grafica.transformations as tr
import grafica.basic_shapes as bs
import grafica.scene_graph as sg
import grafica.easy_shaders as es
import grafica.lighting_shaders as ls
import grafica.performance_monitor as pm
import grafica.text_renderer as tx
from grafica.assets_path import getAssetPath
from operator import add

__author__ = "Ivan Sipiran"
__license__ = "MIT"
#offset inicial
offset_cam=1.5
# A class to store the application control
class Controller:
    def __init__(self):
        #propiedades de los poligonos y ejes
        self.fillPolygon = True
        self.showAxis = True

        #altura de la camara, es como un offset para la esfera 
        #que va a estar dando vueltas con la camara
        self.height=1.6

        #posicion inciial de la camara
        self.viewPos = np.array([2.0, 0.5, 5.0+offset_cam])

        #donde mira inicialmente(al auto)
        self.at = np.array([2.0, -0.037409, 5.0])

        #Eje de referencia
        self.camUp = np.array([0, 1, 0])

        #venia con el programa
        self.distance = 20
        
        #radio de la esfera de la camara
        self.radius= 1.99
        
        #parametro velocidad del auto
        self.carSpeed=0.009

        #angulo inciial del auto c/r al ¡¡ eje Z DEL AUTO !!
        self.theta = 0.0

        #posicion inicial de auto
        self.carPos = np.array([2.0, -0.037409, 5.0])



controller = Controller()

def setPlot(texPipeline, axisPipeline, lightPipeline):
    projection = tr.perspective(45, float(width)/float(height), 0.1, 100)

    glUseProgram(axisPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(axisPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

    glUseProgram(texPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(texPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

    glUseProgram(lightPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(lightPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
    
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "La"), 1.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "Ka"), 0.2, 0.2, 0.2)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "Kd"), 0.9, 0.9, 0.9)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "Ks"), 1.0, 1.0, 1.0)

    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "lightPosition"), 5, 5, 5)
    
    glUniform1ui(glGetUniformLocation(lightPipeline.shaderProgram, "shininess"), 1000)
    glUniform1f(glGetUniformLocation(lightPipeline.shaderProgram, "constantAttenuation"), 0.1)
    glUniform1f(glGetUniformLocation(lightPipeline.shaderProgram, "linearAttenuation"), 0.1)
    glUniform1f(glGetUniformLocation(lightPipeline.shaderProgram, "quadraticAttenuation"), 0.01)

def setView(texPipeline, axisPipeline, lightPipeline):
    global controller
    
    
    view = tr.lookAt(
            controller.viewPos,
            controller.at,
            controller.camUp
        )

    glUseProgram(axisPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(axisPipeline.shaderProgram, "view"), 1, GL_TRUE, view)

    glUseProgram(texPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(texPipeline.shaderProgram, "view"), 1, GL_TRUE, view)

    glUseProgram(lightPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(lightPipeline.shaderProgram, "view"), 1, GL_TRUE, view)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "viewPosition"), controller.viewPos[0], controller.viewPos[1], controller.viewPos[2])
    

def on_key(window, key, scancode, action, mods):

    if action != glfw.PRESS:
        return
    
    global controller

    if key == glfw.KEY_SPACE:
        controller.fillPolygon = not controller.fillPolygon

    elif key == glfw.KEY_LEFT_CONTROL:
        controller.showAxis = not controller.showAxis

    elif key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)
    
    elif key == glfw.KEY_W:
        controller.carPos -= controller.carSpeed * np.array([np.cos(controller.theta),0,np.sin(controller.theta)])

    elif key == glfw.KEY_S:
        controller.carPos += controller.carSpeed * np.array([np.cos(controller.theta),0,np.sin(controller.theta)])
    
    elif key == glfw.KEY_D:
        controller.theta -= 0.001

    elif key == glfw.KEY_A:
        controller.carPos += 0.001
    
    elif key == glfw.KEY_1:
        controller.viewPos = np.array([controller.distance,controller.distance,controller.distance]) #Vista diagonal 1
        controller.camUp = np.array([0,1,0])
    
    elif key == glfw.KEY_2:
        controller.viewPos = np.array([0,0,controller.distance]) #Vista frontal
        controller.camUp = np.array([0,1,0])

    elif key == glfw.KEY_3:
        controller.viewPos = np.array([controller.distance,0,controller.distance]) #Vista lateral
        controller.camUp = np.array([0,1,0])

    elif key == glfw.KEY_4:
        controller.viewPos = np.array([0,controller.distance,0]) #Vista superior
        controller.camUp = np.array([1,0,0])
    
    elif key == glfw.KEY_5:
        controller.viewPos = np.array([controller.distance,controller.distance,-controller.distance]) #Vista diagonal 2
        controller.camUp = np.array([0,1,0])
    
    elif key == glfw.KEY_6:
        controller.viewPos = np.array([-controller.distance,controller.distance,-controller.distance]) #Vista diagonal 2
        controller.camUp = np.array([0,1,0])
    
    elif key == glfw.KEY_7:
        controller.viewPos = np.array([-controller.distance,controller.distance,controller.distance]) #Vista diagonal 2
        controller.camUp = np.array([0,1,0])
    
    else:
        print('Unknown key')

def createOFFShape(pipeline, filename, r,g, b):
    shape = readOFF(getAssetPath(filename), (r, g, b))
    gpuShape = es.GPUShape().initBuffers()
    pipeline.setupVAO(gpuShape)
    gpuShape.fillBuffers(shape.vertices, shape.indices, GL_STATIC_DRAW)

    return gpuShape

def readOFF(filename, color):
    vertices = []
    normals= []
    faces = []

    with open(filename, 'r') as file:
        line = file.readline().strip()
        assert line=="OFF"

        line = file.readline().strip()
        aux = line.split(' ')

        numVertices = int(aux[0])
        numFaces = int(aux[1])

        for i in range(numVertices):
            aux = file.readline().strip().split(' ')
            vertices += [float(coord) for coord in aux[0:]]
        
        vertices = np.asarray(vertices)
        vertices = np.reshape(vertices, (numVertices, 3))
        print(f'Vertices shape: {vertices.shape}')

        normals = np.zeros((numVertices,3), dtype=np.float32)
        print(f'Normals shape: {normals.shape}')

        for i in range(numFaces):
            aux = file.readline().strip().split(' ')
            aux = [int(index) for index in aux[0:]]
            faces += [aux[1:]]
            
            vecA = [vertices[aux[2]][0] - vertices[aux[1]][0], vertices[aux[2]][1] - vertices[aux[1]][1], vertices[aux[2]][2] - vertices[aux[1]][2]]
            vecB = [vertices[aux[3]][0] - vertices[aux[2]][0], vertices[aux[3]][1] - vertices[aux[2]][1], vertices[aux[3]][2] - vertices[aux[2]][2]]

            res = np.cross(vecA, vecB)
            normals[aux[1]][0] += res[0]  
            normals[aux[1]][1] += res[1]  
            normals[aux[1]][2] += res[2]  

            normals[aux[2]][0] += res[0]  
            normals[aux[2]][1] += res[1]  
            normals[aux[2]][2] += res[2]  

            normals[aux[3]][0] += res[0]  
            normals[aux[3]][1] += res[1]  
            normals[aux[3]][2] += res[2]  
        #print(faces)
        norms = np.linalg.norm(normals,axis=1)
        normals = normals/norms[:,None]

        color = np.asarray(color)
        color = np.tile(color, (numVertices, 1))

        vertexData = np.concatenate((vertices, color), axis=1)
        vertexData = np.concatenate((vertexData, normals), axis=1)

        print(vertexData.shape)

        indices = []
        vertexDataF = []
        index = 0

        for face in faces:
            vertex = vertexData[face[0],:]
            vertexDataF += vertex.tolist()
            vertex = vertexData[face[1],:]
            vertexDataF += vertex.tolist()
            vertex = vertexData[face[2],:]
            vertexDataF += vertex.tolist()
            
            indices += [index, index + 1, index + 2]
            index += 3        



        return bs.Shape(vertexDataF, indices)

def createGPUShape(pipeline, shape):
    gpuShape = es.GPUShape().initBuffers()
    pipeline.setupVAO(gpuShape)
    gpuShape.fillBuffers(shape.vertices, shape.indices, GL_STATIC_DRAW)

    return gpuShape

def createTexturedArc(d):
    vertices = [d, 0.0, 0.0, 0.0, 0.0,
                d+1.0, 0.0, 0.0, 1.0, 0.0]
    
    currentIndex1 = 0
    currentIndex2 = 1

    indices = []

    cont = 1
    cont2 = 1

    for angle in range(4, 185, 5):
        angle = np.radians(angle)
        rot = tr.rotationY(angle)
        p1 = rot.dot(np.array([[d],[0],[0],[1]]))
        p2 = rot.dot(np.array([[d+1],[0],[0],[1]]))

        p1 = np.squeeze(p1)
        p2 = np.squeeze(p2)
        
        vertices.extend([p2[0], p2[1], p2[2], 1.0, cont/4])
        vertices.extend([p1[0], p1[1], p1[2], 0.0, cont/4])
        
        indices.extend([currentIndex1, currentIndex2, currentIndex2+1])
        indices.extend([currentIndex2+1, currentIndex2+2, currentIndex1])

        if cont > 4:
            cont = 0


        vertices.extend([p1[0], p1[1], p1[2], 0.0, cont/4])
        vertices.extend([p2[0], p2[1], p2[2], 1.0, cont/4])

        currentIndex1 = currentIndex1 + 4
        currentIndex2 = currentIndex2 + 4
        cont2 = cont2 + 1
        cont = cont + 1

    return bs.Shape(vertices, indices)

def createTiledFloor(dim):
    vert = np.array([[-0.5,0.5,0.5,-0.5],[-0.5,-0.5,0.5,0.5],[0.0,0.0,0.0,0.0],[1.0,1.0,1.0,1.0]], np.float32)
    rot = tr.rotationX(-np.pi/2)
    vert = rot.dot(vert)

    indices = [
         0, 1, 2,
         2, 3, 0]

    vertFinal = []
    indexFinal = []
    cont = 0

    for i in range(-dim,dim,1):
        for j in range(-dim,dim,1):
            tra = tr.translate(i,0.0,j)
            newVert = tra.dot(vert)

            v = newVert[:,0][:-1]
            vertFinal.extend([v[0], v[1], v[2], 0, 1])
            v = newVert[:,1][:-1]
            vertFinal.extend([v[0], v[1], v[2], 1, 1])
            v = newVert[:,2][:-1]
            vertFinal.extend([v[0], v[1], v[2], 1, 0])
            v = newVert[:,3][:-1]
            vertFinal.extend([v[0], v[1], v[2], 0, 0])
            
            ind = [elem + cont for elem in indices]
            indexFinal.extend(ind)
            cont = cont + 4

    return bs.Shape(vertFinal, indexFinal)

# TAREA3: Implementa la función "createHouse" que crea un objeto que representa una casa
# y devuelve un nodo de un grafo de escena (un objeto sg.SceneGraphNode) que representa toda la geometría y las texturas
# Esta función recibe como parámetro el pipeline que se usa para las texturas (texPipeline)

def createHouse(pipeline,wallTexture,roofTexture):
    #primitiva murallas de la casa
    aWallShape = createGPUShape(pipeline, bs.createTextureQuad(1.0, 1.0))
    aWallShape.texture = es.textureSimpleSetup(
        getAssetPath(wallTexture), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST)
    glGenerateMipmap(GL_TEXTURE_2D)

    #primitiva techo de la casa
    aRoofShape = createGPUShape(pipeline, bs.createTextureQuad(1.0, 1.0))
    aRoofShape.texture = es.textureSimpleSetup(
        getAssetPath(roofTexture), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST)
    glGenerateMipmap(GL_TEXTURE_2D)
    
    #-------- murallas de la casa------------
    #pared rotada
    wallHouseNodeRotated = sg.SceneGraphNode('murallaHouse')
    wallHouseNodeRotated.transform = tr.matmul([tr.scale(1,1,2),tr.rotationY(np.pi/2)])
    wallHouseNodeRotated.childs +=[aWallShape]

    #paredes paralelo al eje X
    #pared en posicion z=1
    wallHouseXPos= sg.SceneGraphNode('murallaPosX')
    wallHouseXPos.transform = tr.translate(0,0,1)
    wallHouseXPos.childs +=[aWallShape]
    #pared en posicion z=1
    wallHouseXNeg= sg.SceneGraphNode('murallaNegX')
    wallHouseXNeg.transform = tr.translate(0,0,-1)
    wallHouseXNeg.childs +=[aWallShape]

    #paredes paralelo al ejeZ
    #pared en posicion x=-0.5
    wallHouseZPos= sg.SceneGraphNode('murallaPosZ')
    wallHouseZPos.transform = tr.translate(-0.5,0,0)
    wallHouseZPos.childs +=[wallHouseNodeRotated]
    #pared en posicion x=0.5
    wallHouseZNeg= sg.SceneGraphNode('murallaPosZ')
    wallHouseZNeg.transform = tr.translate(0.5,0,0)
    wallHouseZNeg.childs +=[wallHouseNodeRotated]

    #cielo de las paredes de la casa
    ceilingHouse = sg.SceneGraphNode('murallaCielo')
    ceilingHouse.transform = tr.matmul([tr.scale(1,1,2),tr.translate(0,0.5,0),tr.rotationX(np.pi/2)])
    ceilingHouse.childs += [aWallShape]

    #nodo de las murallas y el cielo
    wallsHouse = sg.SceneGraphNode('wallsHouse')
    wallsHouse.childs += [wallHouseXPos,wallHouseXNeg,
                    wallHouseZNeg,wallHouseZPos,ceilingHouse] 
    
    #------------termino murallas y cielo--------------

    #----------techo de la casa-------------
    #primero creemos un cubo centrado para poder asi rotarlo y hacer cosas

    CubeXPos= sg.SceneGraphNode('murallaPosX')
    CubeXPos.transform = tr.translate(0,0,0.5)
    CubeXPos.childs +=[aRoofShape]

    CubeXNeg= sg.SceneGraphNode('murallaNegX')
    CubeXNeg.transform = tr.translate(0,0,-0.5)
    CubeXNeg.childs +=[aRoofShape]

    CubeZPos= sg.SceneGraphNode('murallaPosZ')
    CubeZPos.transform = tr.matmul([tr.translate(0.5,0,0),tr.rotationY(np.pi/2)])
    CubeZPos.childs +=[aRoofShape]
   
    CubeZNeg= sg.SceneGraphNode('murallaPosZ')
    CubeZNeg.transform = tr.matmul([tr.translate(-0.5,0,0),tr.rotationY(np.pi/2)])
    CubeZNeg.childs +=[aRoofShape]

    CubeYPos= sg.SceneGraphNode('murallaPosZ')
    CubeYPos.transform = tr.matmul([tr.translate(0,0.5,0),tr.rotationX(np.pi/2)])
    CubeYPos.childs +=[aRoofShape]
   
    CubeYNeg= sg.SceneGraphNode('murallaPosZ')
    CubeYNeg.transform = tr.matmul([tr.translate(0,-0.5,0),tr.rotationX(np.pi/2)])
    CubeYNeg.childs +=[aRoofShape]

    cube=sg.SceneGraphNode('cubo de textura')
    cube.childs +=[CubeXNeg,CubeXPos,CubeYNeg,CubeYPos,CubeZPos,CubeZNeg]

    #creado el cubo ahora podemos hacer un techo con consistencia

    roofLeft=sg.SceneGraphNode('techoLeft')
    roofLeft.transform = tr.matmul([tr.translate(-0.25,0.75,0),tr.rotationZ(np.pi/4),tr.scale(0.75,1/16,2)])
    roofLeft.childs += [cube]

    roofRight=sg.SceneGraphNode('techoRight')
    roofRight.transform = tr.matmul([tr.translate(0.25,0.75,0),tr.rotationZ(-np.pi/4),tr.scale(0.75,1/16,2)])
    roofRight.childs += [cube]

    roofNode=sg.SceneGraphNode('techo')
    roofNode.childs += [roofLeft,roofRight]


    #------------termino techo de la casa---------
    
    scene = sg.SceneGraphNode('world')
    scene.childs += [roofNode,wallsHouse] 
    return scene

# TAREA3: Implementa la función "createWall" que crea un objeto que representa un muro
# y devuelve un nodo de un grafo de escena (un objeto sg.SceneGraphNode) que representa toda la geometría y las texturas
# Esta función recibe como parámetro el pipeline que se usa para las texturas (texPipeline)

def createWall(pipeline,wallTexture):

    aWallShape = createGPUShape(pipeline, bs.createTextureQuad(1.0, 1.0))
    aWallShape.texture = es.textureSimpleSetup(
        getAssetPath(wallTexture), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST)
    glGenerateMipmap(GL_TEXTURE_2D)

    #primero creemos un cubo centrado para poder asi rotarlo y hacer cosas
    
    CubeXPos= sg.SceneGraphNode('murallaPosX')
    CubeXPos.transform = tr.translate(0,0,0.5)
    CubeXPos.childs +=[aWallShape]

    CubeXNeg= sg.SceneGraphNode('murallaNegX')
    CubeXNeg.transform = tr.translate(0,0,-0.5)
    CubeXNeg.childs +=[aWallShape]

    CubeZPos= sg.SceneGraphNode('murallaPosZ')
    CubeZPos.transform = tr.matmul([tr.translate(0.5,0,0),tr.rotationY(np.pi/2)])
    CubeZPos.childs +=[aWallShape]
   
    CubeZNeg= sg.SceneGraphNode('murallaPosZ')
    CubeZNeg.transform = tr.matmul([tr.translate(-0.5,0,0),tr.rotationY(np.pi/2)])
    CubeZNeg.childs +=[aWallShape]

    CubeYPos= sg.SceneGraphNode('murallaPosZ')
    CubeYPos.transform = tr.matmul([tr.translate(0,0.5,0),tr.rotationX(np.pi/2)])
    CubeYPos.childs +=[aWallShape]
   
    CubeYNeg= sg.SceneGraphNode('murallaPosZ')
    CubeYNeg.transform = tr.matmul([tr.translate(0,-0.5,0),tr.rotationX(np.pi/2)])
    CubeYNeg.childs +=[aWallShape]

    
    cube=sg.SceneGraphNode('cubo de textura')
    cube.childs +=[CubeXNeg,CubeXPos,CubeYNeg,CubeYPos,CubeZPos,CubeZNeg]

    #luego podemos hacer las murallitas de contencion en filas
    aBarrier=sg.SceneGraphNode('barrerita')
    aBarrier.transform = tr.matmul([tr.scale(1/16,1/5,1)])
    aBarrier.childs += [cube]

    scene=sg.SceneGraphNode('world')
    scene.childs += [aBarrier]

    return scene

# TAREA3: Esta función crea un grafo de escena especial para el auto.
def createCarScene(pipeline):
    chasis = createOFFShape(pipeline, 'alfa2.off', 1.0, 0.0, 0.0)
    wheel = createOFFShape(pipeline, 'wheel.off', 0.0, 0.0, 0.0)

    scale = 2.0
    rotatingWheelNode = sg.SceneGraphNode('rotatingWheel')
    rotatingWheelNode.childs += [wheel]

    chasisNode = sg.SceneGraphNode('chasis')
    chasisNode.transform = tr.uniformScale(scale)
    chasisNode.childs += [chasis]

    wheel1Node = sg.SceneGraphNode('wheel1')
    wheel1Node.transform = tr.matmul([tr.uniformScale(scale),tr.translate(0.056390,0.037409,0.091705)])
    wheel1Node.childs += [rotatingWheelNode]

    wheel2Node = sg.SceneGraphNode('wheel2')
    wheel2Node.transform = tr.matmul([tr.uniformScale(scale),tr.translate(-0.060390,0.037409,-0.091705)])
    wheel2Node.childs += [rotatingWheelNode]

    wheel3Node = sg.SceneGraphNode('wheel3')
    wheel3Node.transform = tr.matmul([tr.uniformScale(scale),tr.translate(-0.056390,0.037409,0.091705)])
    wheel3Node.childs += [rotatingWheelNode]

    wheel4Node = sg.SceneGraphNode('wheel4')
    wheel4Node.transform = tr.matmul([tr.uniformScale(scale),tr.translate(0.066090,0.037409,-0.091705)])
    wheel4Node.childs += [rotatingWheelNode]

    car1 = sg.SceneGraphNode('car1')
    car1.transform = tr.matmul([tr.translate(2.0, -0.037409, 5.0), tr.rotationY(np.pi)])
    car1.childs += [chasisNode]
    car1.childs += [wheel1Node]
    car1.childs += [wheel2Node]
    car1.childs += [wheel3Node]
    car1.childs += [wheel4Node]

    scene = sg.SceneGraphNode('system')
    scene.childs += [car1]

    return scene

# TAREA3: Esta función crea toda la escena estática y texturada de esta aplicación.
# Por ahora ya están implementadas: la pista y el terreno
# En esta función debes incorporar las casas y muros alrededor de la pista

def createStaticScene(pipeline):

    roadBaseShape = createGPUShape(pipeline, bs.createTextureQuad(1.0, 1.0))
    roadBaseShape.texture = es.textureSimpleSetup(
        getAssetPath("Road_001_basecolor.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST)
    glGenerateMipmap(GL_TEXTURE_2D)

    sandBaseShape = createGPUShape(pipeline, createTiledFloor(50))
    sandBaseShape.texture = es.textureSimpleSetup(
        getAssetPath("Sand 002_COLOR.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST)
    glGenerateMipmap(GL_TEXTURE_2D)

    arcShape = createGPUShape(pipeline, createTexturedArc(1.5))
    arcShape.texture = roadBaseShape.texture
    
    roadBaseNode = sg.SceneGraphNode('plane')
    roadBaseNode.transform = tr.rotationX(-np.pi/2)
    roadBaseNode.childs += [roadBaseShape]

    arcNode = sg.SceneGraphNode('arc')
    arcNode.childs += [arcShape]

    sandNode = sg.SceneGraphNode('sand')
    sandNode.transform = tr.translate(0.0,-0.1,0.0)
    sandNode.childs += [sandBaseShape]

    linearSector = sg.SceneGraphNode('linearSector')
        
    for i in range(10):
        node = sg.SceneGraphNode('road'+str(i)+'_ls')
        node.transform = tr.translate(0.0,0.0,-1.0*i)
        node.childs += [roadBaseNode]
        linearSector.childs += [node]

    linearSectorLeft = sg.SceneGraphNode('lsLeft')
    linearSectorLeft.transform = tr.translate(-2.0, 0.0, 5.0)
    linearSectorLeft.childs += [linearSector]

    linearSectorRight = sg.SceneGraphNode('lsRight')
    linearSectorRight.transform = tr.translate(2.0, 0.0, 5.0)
    linearSectorRight.childs += [linearSector]

    arcTop = sg.SceneGraphNode('arcTop')
    arcTop.transform = tr.translate(0.0,0.0,-4.5)
    arcTop.childs += [arcNode]

    arcBottom = sg.SceneGraphNode('arcBottom')
    arcBottom.transform = tr.matmul([tr.translate(0.0,0.0,5.5), tr.rotationY(np.pi)])
    arcBottom.childs += [arcNode]
    
    scene = sg.SceneGraphNode('system')
    scene.childs += [linearSectorLeft]
    scene.childs += [linearSectorRight]
    scene.childs += [arcTop]
    scene.childs += [arcBottom]
    scene.childs += [sandNode]
    
    #aca ponemos las casas las casas
    casa1=createHouse(pipeline,"wall4.jpg","roof1.jpg")
    casa1.transform = tr.matmul([tr.translate(0,0,5),tr.uniformScale(0.5)])

    casa2=createHouse(pipeline,"wall2.jpg","roof4.jpg")
    casa2.transform = tr.matmul([tr.translate(0,0,0),tr.uniformScale(0.5)])

    casa3=createHouse(pipeline,"wall1.jpg","roof4.jpg")
    casa3.transform = tr.matmul([tr.translate(0,0,-5),tr.uniformScale(0.5)])

    casa4=createHouse(pipeline,"wall2.jpg","roof1.jpg")
    casa4.transform = tr.matmul([tr.translate(4,0,1),tr.uniformScale(0.5)])

    casa5=createHouse(pipeline,"wall1.jpg","roof4.jpg")
    casa5.transform = tr.matmul([tr.translate(7,0,-1),tr.uniformScale(0.5)])

    casa6=createHouse(pipeline,"wall3.jpg","roof4.jpg")
    casa6.transform = tr.matmul([tr.translate(5,0,3),tr.uniformScale(0.5)])

    casa7=createHouse(pipeline,"wall4.jpg","roof1.jpg")
    casa7.transform = tr.matmul([tr.translate(-4,0,1),tr.uniformScale(0.5)])

    casa8=createHouse(pipeline,"wall4.jpg","roof4.jpg")
    casa8.transform = tr.matmul([tr.translate(-7,0,-1),tr.uniformScale(0.5)])

    casa9=createHouse(pipeline,"wall3.jpg","roof4.jpg")
    casa9.transform = tr.matmul([tr.translate(-5,0,3),tr.uniformScale(0.5)])

    scene.childs += [casa1,casa2,casa3,casa4,casa5,casa6,casa7,casa8,casa9]

    #veamos la barrera
    #al ojimtero creamos una barrera cerca de cada camino
    for i in range(9):
        barrera = createWall(pipeline, "wall5.jpg")
        barrera.transform = tr.translate(2.55,0,1*i-3)

        barrera2 = createWall(pipeline, "wall5.jpg")
        barrera2.transform = tr.translate(1.45,0,1*i-3)

        barrera3 = createWall(pipeline, "wall5.jpg")
        barrera3.transform = tr.translate(-1.45,0,1*i-3)

        barrera4 = createWall(pipeline, "wall5.jpg")
        barrera4.transform = tr.translate(-2.55,0,1*i-3)

        scene.childs += [barrera, barrera2, barrera3, barrera4]
    
    return scene

if __name__ == "__main__":

    # Initialize glfw
    if not glfw.init():
        glfw.set_window_should_close(window, True)

    width = 800
    height = 800
    title = "Tarea 2"
    window = glfw.create_window(width, height, title, None, None)

    if not window:
        glfw.terminate()
        glfw.set_window_should_close(window, True)

    glfw.make_context_current(window)

    # Connecting the callback function 'on_key' to handle keyboard events
    glfw.set_key_callback(window, on_key)

    # Assembling the shader program (pipeline) with both shaders
    axisPipeline = es.SimpleModelViewProjectionShaderProgram()
    texPipeline = es.SimpleTextureModelViewProjectionShaderProgram()
    lightPipeline = ls.SimpleGouraudShaderProgram()
    textPipeline = tx.TextureTextRendererShaderProgram()
    # Telling OpenGL to use our shader program
    glUseProgram(axisPipeline.shaderProgram)

    # Creating texture with all characters
    textBitsTexture = tx.generateTextBitsTexture()
    # Moving texture to GPU memory
    gpuText3DTexture = tx.toOpenGLTexture(textBitsTexture)

    # Setting up the clear screen color
    glClearColor(0.85, 0.85, 0.85, 1.0)

    # As we work in 3D, we need to check which part is in front,
    # and which one is at the back
    glEnable(GL_DEPTH_TEST)

    # Creating shapes on GPU memory
    cpuAxis = bs.createAxis(7)
    gpuAxis = es.GPUShape().initBuffers()
    axisPipeline.setupVAO(gpuAxis)
    gpuAxis.fillBuffers(cpuAxis.vertices, cpuAxis.indices, GL_STATIC_DRAW)

    #espacio para el texto!--------------
    speedCharSize = 0.15
    speedShape=tx.textToShape(str(controller.carSpeed),speedCharSize,speedCharSize)
    gpuSpeed = es.GPUShape().initBuffers()
    textPipeline.setupVAO(gpuSpeed)
    gpuSpeed.fillBuffers(speedShape.vertices, speedShape.indices, GL_STATIC_DRAW)
    gpuSpeed.texture = gpuText3DTexture
    #-------------------

    #NOTA: Aqui creas un objeto con tu escena
    dibujo = createStaticScene(texPipeline)
    car = createCarScene(lightPipeline)

    setPlot(texPipeline, axisPipeline,lightPipeline)

    perfMonitor = pm.PerformanceMonitor(glfw.get_time(), 0.5)
    t0 = glfw.get_time()
    # glfw will swap buffers as soon as possible
    glfw.swap_interval(0)

    while not glfw.window_should_close(window):

        # Measuring performance
        perfMonitor.update(glfw.get_time())
        glfw.set_window_title(window, title + str(perfMonitor))

        # Using GLFW to check for input events
        glfw.poll_events()

        # Getting the time difference from the previous iteration
        t1 = glfw.get_time()
        dt = t1 - t0
        t0 = t1

        #angulos para avanzar y retroceder respectivamente.
        #indica la latitud de donde se mira con la camara
        phi1=-0.5; phi2=0.5
        
        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
            if controller.carSpeed<0:
                controller.carSpeed += dt/10
            #se actualiza la velocidad; 0.11 se encontro que al ojimetro era muy rapido y se setio como el limite de velocidad
            if controller.carSpeed <=0.10:
                controller.carSpeed += dt/100
            #si avanza se actualiza la posicion del auto
            controller.carPos -= controller.carSpeed*np.array([np.sin(controller.theta),0,np.cos(controller.theta)])

            #si doblamos se actualiza el angulo
            if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
                controller.theta += 2*dt
            
            if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
                controller.theta -= 2*dt
            
            #ahora situamos las coordenadas esfericas de la camara
            camX = controller.radius * np.sin(controller.theta) * np.cos(phi1)
            camY = controller.radius * np.sin(phi1)
            camZ = controller.radius * np.cos(controller.theta) * np.cos(phi1)
            
            #notemos que la posicion de vista es:
            #el vector donde esta el auto, un offset que es la altura del centro de la esfera y la esfera en si
            controller.viewPos = controller.carPos +np.array([0,controller.height,0])+ np.array([camX,camY,camZ])
            
            #ahora le decimos que mire el auto como referencia
            controller.at = controller.carPos 
        #si no precionamos la tecla

        #aca es excatamente lo mismo a lo anterior
        elif glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            if controller.carSpeed>0:
                controller.carSpeed -= dt/50
            
            if controller.carSpeed >=- 0.05:
                 controller.carSpeed -= dt/100
            
            controller.carPos -= controller.carSpeed*np.array([np.sin(controller.theta),0,np.cos(controller.theta)])

            if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
                controller.theta -= 2*dt

            if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
                controller.theta += 2*dt
            
            #es la misma dinamica que el anterior pero para retroceder!
            #pero seguimos situando la camara en la misma esfera
            camX = controller.radius * np.sin(controller.theta) * np.cos(-phi1)
            camY = controller.radius * np.sin(-phi1)
            camZ = controller.radius * np.cos(controller.theta) * np.cos(-phi1)
            
            controller.viewPos = controller.carPos +np.array([0,controller.height,0]) - np.array([camX,camY,camZ])
            
            controller.at = controller.carPos
        else:
            #si la velocidad es <0  la "aumentamos"
            if(controller.carSpeed < 0):
                controller.carSpeed = min(controller.carSpeed+dt/8, 0)
                #si la velocidad es <0 aun podemos doblar el auto
                if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
                    controller.theta -= 2*dt

                if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
                    controller.theta += 2*dt
                #actualizamos la posicion del auto
                controller.carPos -= controller.carSpeed*np.array([np.sin(controller.theta),0,np.cos(controller.theta)])
                #actualizamos la pos de la camara en la esfera
                camX = controller.radius * np.sin(controller.theta) * np.cos(-phi1)
                camY = controller.radius * np.sin(-phi1)
                camZ = controller.radius * np.cos(controller.theta) * np.cos(-phi1)
                #actualizamos la posicion de la camara
                controller.viewPos = controller.carPos +np.array([0,controller.height,0]) - np.array([camX,camY,camZ])
            #si la velocidad es > 0 la reducimos
            elif(controller.carSpeed > 0):
                controller.carSpeed = max(controller.carSpeed-dt/8, 0)
                #si la velocidad es >0 aun podemos doblar el auto
                if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
                    controller.theta += 2*dt
            
                if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
                    controller.theta -= 2*dt
                #actualizamos la posicion del auto
                controller.carPos -= controller.carSpeed*np.array([np.sin(controller.theta),0,np.cos(controller.theta)])
                #actualizamos la pos de la camara en la esfera
                camX = controller.radius * np.sin(controller.theta) * np.cos(phi1)
                camY = controller.radius * np.sin(phi1)
                camZ = controller.radius * np.cos(controller.theta) * np.cos(phi1)
                #actualizamos la posicion de la camara
                controller.viewPos = controller.carPos +np.array([0,controller.height,0])+ np.array([camX,camY,camZ])



        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Filling or not the shapes depending on the controller state
        if (controller.fillPolygon):
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        setView(texPipeline, axisPipeline, lightPipeline)

        if controller.showAxis:
            glUseProgram(axisPipeline.shaderProgram)
            glUniformMatrix4fv(glGetUniformLocation(axisPipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
            axisPipeline.drawCall(gpuAxis, GL_LINES)
        
        

        #NOTA: Aquí dibujas tu objeto de escena
        glUseProgram(texPipeline.shaderProgram)
        sg.drawSceneGraphNode(dibujo, texPipeline, "model")

        glUseProgram(lightPipeline.shaderProgram)
        #actualizamos el auto; buscamos el nodo, luego le decimos que cambie su transformacion a que primero rote pi+theta
        #y luego lo transladamos como antes por la misma posicion!
        sg.findNode(car,"car1").transform = tr.matmul([tr.translate(controller.carPos[0],controller.carPos[1],controller.carPos[2]),
                                                        tr.rotationY(np.pi+controller.theta)])
        sg.drawSceneGraphNode(car, lightPipeline, "model")

        

        # Once the render is done, buffers are swapped, showing only the complete scene.
        glfw.swap_buffers(window)

    # freeing GPU memory
    gpuAxis.clear()
    dibujo.clear()
    

    glfw.terminate()