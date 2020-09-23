#version 450
uniform mat4 V;
uniform mat4 P;
uniform mat4 pose_rot;
uniform mat4 pose_trans;
uniform vec3 instance_color;
uniform vec3 diffuse_color;
uniform float max_length;
uniform float z_offset;

layout (location=0) in vec3 position;
layout (location=1) in vec3 normal;
layout (location=2) in vec2 texCoords;
out vec2 theCoords;
out vec3 Normal;
out vec3 FragPos;
out vec3 Normal_cam;
out vec3 Instance_color;
out vec3 Pos_cam;
out vec3 Diffuse_color;
void main() {
    vec4 temp_pos =  V * pose_trans * pose_rot * vec4(position, 1);
    gl_Position.x = temp_pos.x / max_length;
    gl_Position.y = -temp_pos.y / max_length;
    gl_Position.z = - temp_pos.z + z_offset;


    vec4 world_position4 = pose_trans * pose_rot * vec4(position, 1);
    FragPos = vec3(world_position4.xyz / world_position4.w); // in world coordinate
    Normal = normalize(mat3(pose_rot) * normal); // in world coordinate
    Normal_cam = normalize(mat3(V) * mat3(pose_rot) * normal); // in camera coordinate
    vec4 pos_cam4 = V * pose_trans * pose_rot * vec4(position, 1);
    Pos_cam = pos_cam4.xyz / pos_cam4.w;
    theCoords = texCoords;
    Instance_color = instance_color;
    Diffuse_color = diffuse_color;
}