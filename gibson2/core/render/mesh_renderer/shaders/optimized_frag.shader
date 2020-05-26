#version 410

#define MAX_ARRAY_SIZE 512

uniform sampler2DArray bigTex;
uniform sampler2DArray smallTex;

uniform TexColorData {
    vec4 tex_data[MAX_ARRAY_SIZE];
    vec4 diffuse_colors[MAX_ARRAY_SIZE];
};

in vec2 theCoords;
in vec3 Normal;
in vec3 Normal_cam;
in vec3 FragPos;
in vec3 Pos_cam;
flat in int Draw_id;

layout (location = 0) out vec4 outputColour;
layout (location = 1) out vec4 NormalColour;
layout (location = 2) out vec4 InstanceColour;
layout (location = 3) out vec4 PCColour;

uniform vec3 light_position;  // in world coordinate
uniform vec3 light_color; // light color

void main() {
    float ambientStrength = 0.2;
    vec3 ambient = ambientStrength * light_color;
    vec3 lightDir = normalize(light_position - FragPos);
    float diff = 0.5 + 0.5 * max(dot(Normal, lightDir), 0.0);
    vec3 diffuse = diff * light_color;
    vec4 curr_tex_data = tex_data[Draw_id];
    int tex_num = int(curr_tex_data.x);
    int tex_layer = int(curr_tex_data.y);
    float instance_color = curr_tex_data.z;

    if (tex_num == -1) {
	outputColour = diffuse_colors[Draw_id] * diff; //diffuse color
    } else if (tex_num == 0) {
        outputColour = texture(bigTex, vec3(theCoords.x, theCoords.y, tex_layer));
    } else if (tex_num == 1) {
	outputColour = texture(smallTex, vec3(theCoords.x, theCoords.y, tex_layer));
    }

    NormalColour =  vec4((Normal_cam + 1) / 2,1);
    InstanceColour = vec4(instance_color, 0, 0, 1);
    PCColour = vec4(Pos_cam,1);
}