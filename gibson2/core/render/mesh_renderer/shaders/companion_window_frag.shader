#version 450
uniform sampler2D texUnit;
in vec2 TexCoords;

out vec4 FragColor;

void main() {
    FragColor = texture(texUnit, TexCoords);
}
