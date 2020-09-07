#version 450

#define MAX_ARRAY_SIZE 512

layout (std140) uniform TransformData {
    mat4 pose_trans_array[MAX_ARRAY_SIZE];
    mat4 pose_rot_array[MAX_ARRAY_SIZE];
};

in int gl_DrawID;

uniform mat4 V;
uniform mat4 P;

uniform vec3 instance_color;
uniform vec3 diffuse_color;

layout (location=0) in vec3 position;
layout (location=1) in vec3 normal;
layout (location=2) in vec2 texCoords;
layout (location=3) in vec3 tangent;
layout (location=4) in vec3 bitangent;


out vec2 theCoords;
out vec3 Normal_world;
out vec3 FragPos;
out vec3 Normal_cam;
out vec3 Instance_color;
out vec3 Pos_cam;
out vec3 Pos_cam_projected;
out vec3 Diffuse_color;
out mat3 TBN;
flat out int Draw_id;

void main() {
    mat4 pose_trans = pose_trans_array[gl_DrawID];
    mat4 pose_rot = transpose(pose_rot_array[gl_DrawID]);
    gl_Position = P * V * pose_trans * pose_rot * vec4(position, 1);
    vec4 world_position4 = pose_trans * pose_rot * vec4(position, 1);
    FragPos = vec3(world_position4.xyz / world_position4.w); // in world coordinate
    Normal_world = normalize(mat3(pose_rot) * normal); // in world coordinate
    Normal_cam = normalize(mat3(V) * mat3(pose_rot) * normal); // in camera coordinate
    vec4 pos_cam4_projected = P * V * pose_trans * pose_rot * vec4(position, 1);
    Pos_cam_projected = pos_cam4_projected.xyz / pos_cam4_projected.w;
    vec4 pos_cam4 = V * pose_trans * pose_rot * vec4(position, 1);
    Pos_cam = pos_cam4.xyz / pos_cam4.w;
    theCoords = texCoords;
    Instance_color = instance_color;
    Diffuse_color = diffuse_color;
    vec3 T = normalize(vec3(pose_trans * pose_rot * vec4(tangent,   0.0)));
    vec3 B = normalize(vec3(pose_trans * pose_rot * vec4(bitangent, 0.0)));
    vec3 N = normalize(vec3(pose_trans * pose_rot * vec4(normal,    0.0)));
    TBN = mat3(T, B, N);
    Draw_id = gl_DrawID;
}