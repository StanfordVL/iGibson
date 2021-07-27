class VisualObject(object):
    """
    A visual object manages a set of VAOs and textures
    A wavefront obj file is loaded into openGL and managed by a VisualObject
    """

    def __init__(self, filename, VAO_ids, vertex_data_indices, face_indices, id, renderer):
        """
        :param filename: filename of the obj file
        :param VAO_ids: VAO_ids in OpenGL
        :param vertex_data_indices: vertex data indices
        :param face_indices: face data indices
        :param id: renderer maintains a list of visual objects, id is the handle of a visual object
        :param renderer: pointer to the renderer
        """
        self.VAO_ids = VAO_ids
        self.filename = filename
        self.texture_ids = []
        self.id = id
        self.renderer = renderer
        self.vertex_data_indices = vertex_data_indices
        self.face_indices = face_indices

    def __str__(self):
        return "Object({})->VAO({})".format(self.id, self.VAO_ids)

    def __repr__(self):
        return self.__str__()
