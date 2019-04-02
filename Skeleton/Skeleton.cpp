//=============================================================================================
// Mintaprogram: Z�ld h�romsz�g. Ervenyes 2018. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : K�lm�n Zsolt
// Neptun : VCNMZR
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL
const char * vertexSource = R"(
	#version 330
    precision highp float;

	uniform mat4 MVP;			// Model-View-Projection matrix in row-major format

	layout(location = 0) in vec2 vertexPosition;	// Attrib Array 0
	layout(location = 1) in vec2 vertexUV;			// Attrib Array 1

	out vec2 texCoord;								// output attribute

	void main() {
		texCoord = vertexUV;														// copy texture coordinates
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * MVP; 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char * fragmentSource = R"(
	#version 330
    precision highp float;
	uniform vec3 color;
	uniform sampler2D textureUnit;
	uniform int isGPUProcedural;

	in vec2 texCoord;			// variable input: interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation


	void main() {
		if(isGPUProcedural != 0){
			fragmentColor = texture(textureUnit, texCoord);
		}else{
			fragmentColor = vec4(color, 1);
		}
	}
)";


GPUProgram gpuProgram; // vertex and fragment shaders
unsigned int vao;	   // virtual world on the GPU

					   // Initialization, create an OpenGL context


					   // 2D camera
					   // 2D camera
class Camera2D {
	vec2 wCenter; // center in world coordinates
	vec2 wSize;   // width and height in world coordinates
public:
	Camera2D() : wCenter(0, 0), wSize(20, 20) { }

	mat4 V() { return TranslateMatrix(-wCenter); }
	mat4 P() { return ScaleMatrix(vec2(2 / wSize.x, 2 / wSize.y)); }

	mat4 Vinv() { return TranslateMatrix(wCenter); }
	mat4 Pinv() { return ScaleMatrix(vec2(wSize.x / 2, wSize.y / 2)); }

	void Zoom(float s) { wSize = wSize * s; }
	void Pan(vec2 t) { wCenter = wCenter + t; }
	void setCenter(vec2 center) {
		wCenter = center;
	}
	vec4 getCenter() {
		return vec4(wCenter.x, wCenter.y, 0.0f, 1.0f);
	}
};
bool isGPUProcedural = false;
Camera2D camera;	// 2D camera
bool animate = true;
float tCurrent = 0;	// current clock in sec
const int nTesselatedVertices = 2000;
const vec4 gravity = vec4(0.0f, -0.001f, 0.0f, 1.0f);
class TexturedQuad {
	unsigned int vao, vbo[2];
	vec2 vertices[4], uvs[4];
	Texture * pTexture;
public:
	TexturedQuad() {
		vertices[0] = vec2(-50, -50); uvs[0] = vec2(0, 0);
		vertices[1] = vec2(50, -50);  uvs[1] = vec2(1, 0);
		vertices[2] = vec2(50, 50);   uvs[2] = vec2(1, 1);
		vertices[3] = vec2(-50, 50);  uvs[3] = vec2(0, 1);
	}
	void Create() {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		glGenBuffers(2, vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);	   // copy to that part of the memory which will be modified 
		// Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed

		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
		glBufferData(GL_ARRAY_BUFFER, sizeof(uvs), uvs, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		// Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed

		int width = 128, height = 128;
		std::vector<vec4> image(width * height);
		for (int y = 0; y < height; y++) {
			float mountainTop = 1.0f / height;
			for (int x = 0; x < width; x++) {
				image[y * width + x] = vec4(0.6f, 0.48f + mountainTop, 0.32f, 1);
				
			}
		}

		pTexture = new Texture(width, height, image);
	}

	void MoveVertex(float cX, float cY) {
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);

		vec4 wCursor4 = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		vec2 wCursor(wCursor4.x, wCursor4.y);

		int closestVertex = 0;
		float distMin = length(vertices[0] - wCursor);
		for (int i = 1; i < 4; i++) {
			float dist = length(vertices[i] - wCursor);
			if (dist < distMin) {
				distMin = dist;
				closestVertex = i;
			}
		}
		vertices[closestVertex] = wCursor;

		// copy data to the GPU
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);	   // copy to that part of the memory which is modified 
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source

		mat4 MVPTransform = camera.V() * camera.P();

		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		MVPTransform.SetUniform(gpuProgram.getId(), "MVP");

		int location = glGetUniformLocation(gpuProgram.getId(), "isGPUProcedural");
		if (location >= 0) glUniform1i(location, true); // set uniform variable MVP to the MVPTransform
		else printf("isGPUProcedural cannot be set\n");

		pTexture->SetUniform(gpuProgram.getId(), "textureUnit");

		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};


// Forrás: https://codereview.stackexchange.com/questions/144586/finding-the-distance-between-two-points-in-c

float distanceBetweenTwoPoints(float x, float y, float a, float b) {
	return sqrtf(pow(x - a, 2) + pow(y - b, 2));
}



class Curve {
	unsigned int vaoCurve, vboCurve;
	unsigned int vaoCtrlPoints, vboCtrlPoints;
	unsigned int vaoAnimatedObject, vboAnimatedObject;
	int currentSize;
	vec4 color;
	std::vector<vec4> linePoints;
protected:
	std::vector<vec4> wCtrlPoints;		// coordinates of control points
	float fill1;
	float fill2;
	float fill3;
	float fill4;

public:

	virtual float getY(float x){
		return 0.0f;
	}

	virtual void setFill(float point1, float point2, float point3, float point4) {
		fill1 = point1;
		fill2 = point2;
		fill3 = point3;
		fill4 = point4;
	}

	virtual void translateAllControlPoints(vec4 translateVector) {
		for (int i = 0; i < wCtrlPoints.size(); i++) {
			wCtrlPoints[i] = wCtrlPoints[i] * TranslateMatrix(vec3(translateVector.x, translateVector.y, 0.0f));
			
		}
	}

	virtual void setColor(vec4 color) {
		this->color = color;
	}

	virtual void setTension(float tension) {};
	virtual float getTension() { return 0.0f; };

	int getPoints() {
		return wCtrlPoints.size();
	}

	Curve() {
		color = vec4(1.0f, 0.5f, 0.0f, 1.0f);
		// Curve
		currentSize = 0;
		glGenVertexArrays(1, &vaoCurve);
		glBindVertexArray(vaoCurve);

		glGenBuffers(1, &vboCurve); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vboCurve);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
									   // Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), NULL); // attribute array, components/attribute, component type, normalize?, stride, offset

																				  // Control Points
		glGenVertexArrays(1, &vaoCtrlPoints);
		glBindVertexArray(vaoCtrlPoints);

		glGenBuffers(1, &vboCtrlPoints); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vboCtrlPoints);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
									   // Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), NULL); // attribute array, components/attribute, component type, normalize?, stride, offset

																				  // Convex Hull
		glEnableVertexAttribArray(0);  // attribute array 0
									   // Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), NULL); // attribute array, components/attribute, component type, normalize?, stride, offset

																				  // Animated Object
		glGenVertexArrays(1, &vaoAnimatedObject);
		glBindVertexArray(vaoAnimatedObject);

		glGenBuffers(1, &vboAnimatedObject); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vboAnimatedObject);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
									   // Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), NULL); // attribute array, components/attribute, component type, normalize?, stride, offset

	}

	bool isAbove(vec4 point) {
		for (int i = 0; i < linePoints.size(); i++) {
			if (linePoints[i].y > point.y) {
				return false;
			}
		}
		return true;
	}



	virtual vec4 r(float t) { return wCtrlPoints[0]; }
	virtual float tStart() { return 0; }
	virtual float tEnd() { return 1; }

	virtual void AddControlPoint(float cX, float cY) {
		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		wCtrlPoints.push_back(wVertex);
	}



	// Returns the selected control point or -1
	int PickControlPoint(float cX, float cY) {
		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		for (unsigned int p = 0; p < wCtrlPoints.size(); p++) {
			if (dot(wCtrlPoints[p] - wVertex, wCtrlPoints[p] - wVertex) < 0.1) return p;
		}
		return -1;
	}

	void MoveControlPoint(int p, float cX, float cY) {
		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		wCtrlPoints[p] = wVertex;
	}

	void Draw() {
		mat4 VPTransform = camera.V() * camera.P();

		VPTransform.SetUniform(gpuProgram.getId(), "MVP");

		int colorLocation = glGetUniformLocation(gpuProgram.getId(), "color");
		int location = glGetUniformLocation(gpuProgram.getId(), "isGPUProcedural");
		if (location >= 0) glUniform1i(location, false); // set uniform variable MVP to the MVPTransform


		if (wCtrlPoints.size() > 0) {	// draw control points
			glBindVertexArray(vaoCtrlPoints);
			glBindBuffer(GL_ARRAY_BUFFER, vboCtrlPoints);
			glBufferData(GL_ARRAY_BUFFER, wCtrlPoints.size() * 4 * sizeof(float), &wCtrlPoints[0], GL_DYNAMIC_DRAW);
			if (colorLocation >= 0) glUniform3f(colorLocation, 1, 1, 0);
			glPointSize(0.01f);
			glDrawArrays(GL_POINTS, 0, wCtrlPoints.size());
		}

		if (wCtrlPoints.size() >= 4) {	// draw curve
			std::vector<float> vertexData;
			vertexData.push_back(fill1);
			vertexData.push_back(fill2);
			for (float i = -10.0f; i <= 10.0f; i+=0.1f) {	// Tessellate
				vec4 wVertex = vec4(i, getY(i), 0.0f, 1.0f);
				vertexData.push_back(wVertex.x);
				vertexData.push_back(wVertex.y);
			}
			vertexData.push_back(fill3);
			vertexData.push_back(fill4);
			// copy data to the GPU
			glBindVertexArray(vaoCurve);
			glBindBuffer(GL_ARRAY_BUFFER, vboCurve);
			glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), &vertexData[0], GL_DYNAMIC_DRAW);
			if (colorLocation >= 0) glUniform3f(colorLocation, color.x, color.y, color.z);
			glDrawArrays(GL_TRIANGLE_FAN, 0, (20.0f / 0.1f)+2);

			if (animate) {



			}
		}
	}

	std::vector<vec4> getLinePoints() {
		if (currentSize < wCtrlPoints.size()) {
			linePoints.clear();
			for (int i = 0; i < nTesselatedVertices; i++) {	// Tessellate
				float tNormalized = (float)i / (nTesselatedVertices - 1);
				float t = tStart() + (tEnd() - tStart()) * tNormalized;
				vec4 wVertex = r(t);
				linePoints.push_back(wVertex);
			}
			currentSize = wCtrlPoints.size();
		}
		return linePoints;
	}

	vec4 getPointAtPosition(float x) {
		vec4 point = linePoints[0];
		for (int i = 0; i < linePoints.size()-1; i++) {
			if (linePoints[i].x < x && linePoints[i + 1].x  > x) {
				return linePoints[i];
			}
		}
	}
};

//Forrás: http://paulbourke.net/geometry/circlesphere/tvoght.c

std::vector<float> circle_circle_intersection(float x0, float y0, float r0,
	float x1, float y1, float r1)
{
	float a, dx, dy, d, h, rx, ry;
	float x2, y2;

	/* dx and dy are the vertical and horizontal distances between
	 * the circle centers.
	 */
	dx = x1 - x0;
	dy = y1 - y0;

	/* Determine the straight-line distance between the centers. */
	//d = sqrt((dy*dy) + (dx*dx));
	d = hypot(dx, dy); // Suggested by Keith Briggs

	/* Check for solvability. */
	if (d > (r0 + r1))
	{
		/* no solution. circles do not intersect. */
		return std::vector<float>() = {1.0f, 1.0f};
	}
	if (d < fabs(r0 - r1))
	{
		/* no solution. one circle is contained in the other */
		return std::vector<float>() = { 1.0f, 1.0f };

	}

	/* 'point 2' is the point where the line through the circle
	 * intersection points crosses the line between the circle
	 * centers.
	 */

	 /* Determine the distance from point 0 to point 2. */
	a = ((r0*r0) - (r1*r1) + (d*d)) / (2.0f * d);

	/* Determine the coordinates of point 2. */
	x2 = x0 + (dx * a / d);
	y2 = y0 + (dy * a / d);

	/* Determine the distance from point 2 to either of the
	 * intersection points.
	 */
	h = sqrtf((r0*r0) - (a*a));

	/* Now determine the offsets of the intersection points from
	 * point 2.
	 */
	rx = -dy * (h / d);
	ry = dx * (h / d);

	/* Determine the absolute intersection points. */

	return std::vector<float>() = { x2 + rx, y2 + ry, x2 - rx,  y2 - ry };

	//*xi = x2 + rx;
	//*xi_prime = x2 - rx;
	//*yi = y2 + ry;
	//*yi_prime = y2 - ry;

}


Curve * curve;
Curve * skyCurve;

class Circle {
	float position;
	unsigned int vaoCtrlPoint, vboCtrlPoint;
	unsigned int vaoAnimatedObject, vboAnimatedObject;
	vec4 middle;
	bool direction_right;
	const float RADIUS = 0.4f;
	vec4 movementVector;
public:
	Circle() {
		movementVector = vec4(0.0f, 0.0f, 0.0f, 1.0f);
		direction_right = true;
		position = 41.0;
		middle = vec4(1.0f, 1.0f, 0.0f, 1.0f);
		glGenVertexArrays(1, &vaoCtrlPoint);
		glBindVertexArray(vaoCtrlPoint);

		glGenBuffers(1, &vboCtrlPoint); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vboCtrlPoint);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
									   // Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), NULL);
	}

	vec2 getCenter() {
		return vec2(middle.x, middle.y);
	}




	void Draw() {

		mat4 VPTransform = camera.V() * camera.P();
		mat4 rungAnimationRotationMatrix;
		VPTransform.SetUniform(gpuProgram.getId(), "MVP");

		int colorLocation = glGetUniformLocation(gpuProgram.getId(), "color");
		int location = glGetUniformLocation(gpuProgram.getId(), "isGPUProcedural");
		if (location >= 0) glUniform1i(location, false); // set uniform variable MVP to the MVPTransform
		std::vector<float> middlePoint;
		std::vector<float> circlePoints;
		std::vector<float> headPoints;
		std::vector<float> rungPoints;
		std::vector<float> bodyPoints;
		std::vector<float> rightLegPoints;
		std::vector<float> leftLegPoints;
		std::vector<vec4> circleVecPoints;
		vec4 perpendicular;


		float rotationSpeed = tCurrent / -5.0f;
		//setup rotation matrix for animation	
		if (animate) {
			float speed = 0.06f;
		
			vec4 derivate = vec4(middle.x + 0.001f, curve->getY(middle.x + 0.001f), 0.0f, 1.0f) - vec4(middle.x, curve->getY(middle.x), 0.0f, 1.0f);
			float derivateLength = sqrtf(derivate.x * derivate.x + derivate.y * derivate.y);
			float gravityForce = (derivate.y/derivateLength) * -0.06f;

			printf("%f \n", derivate.y / derivateLength);

			if (middle.x > 9.5f) {
				this->direction_right = false;

			}
			else if (middle.x < -9.5f) {
				this->direction_right = true;

			}

			if (direction_right) {
				speed += gravityForce;
				middle.x += speed;
			}

			else {
				speed -= gravityForce;
				middle.x -= speed;
				rotationSpeed *= -1.0f;
			}

			middle.y = curve->getY(middle.x + speed);

			perpendicular = vec4(-derivate.y, derivate.x, 0.0f, 1.0f);
			float length = sqrtf(perpendicular.x * perpendicular.x + perpendicular.y * perpendicular.y);
			perpendicular = (perpendicular / length) * (RADIUS + 0.15f);
			glLineWidth(1.5f);

			rungAnimationRotationMatrix = RotationMatrix(2.0f * 3.1415926f * rotationSpeed, vec3(0, 0, 1));
		}

		middlePoint.push_back(middle.x + perpendicular.x);
		middlePoint.push_back(middle.y + perpendicular.y);

		//draw middle
		glBindVertexArray(vaoCtrlPoint);
		glBindBuffer(GL_ARRAY_BUFFER, vboCtrlPoint);
		glBufferData(GL_ARRAY_BUFFER, middlePoint.size() * sizeof(float), &middlePoint[0], GL_DYNAMIC_DRAW);
		if (colorLocation >= 0) glUniform3f(colorLocation, 1, 0, 0);
		glPointSize(2.0f);
		glDrawArrays(GL_POINTS, 0, 1);

		//draw circle around middle
		const int TESSELATED_VERTICES = 20;
		vec4 circlePoint = vec4(middle.x, middle.y, 0, 1) * TranslateMatrix(vec3(RADIUS, RADIUS, 0.0f));
		circlePoints.push_back(circlePoint.x + perpendicular.x);
		circlePoints.push_back(circlePoint.y + perpendicular.y);
		for (int i = 0; i < TESSELATED_VERTICES + 1; i++) {
			mat4 rotationMatrix = RotationMatrix(2.0f * 3.1415926f * (float)i / (float)TESSELATED_VERTICES, vec3(0, 0, 1));
			vec4 rotatedPoint = circlePoint * TranslateMatrix(vec3(-middle.x, -middle.y, 0)) * rotationMatrix * TranslateMatrix(vec3(middle.x, middle.y, 0));
			circleVecPoints.push_back(rotatedPoint);
			circlePoints.push_back(rotatedPoint.x + perpendicular.x);
			circlePoints.push_back(rotatedPoint.y + perpendicular.y);
		}
		glBufferData(GL_ARRAY_BUFFER, circlePoints.size() * sizeof(float), &circlePoints[0], GL_DYNAMIC_DRAW);
		glDrawArrays(GL_LINE_STRIP, 0, TESSELATED_VERTICES + 2);



		//draw rungs
		const int RUNG_NUMBER = 6;
		vec4 rungPoint;
		rungPoint = vec4(0, 0, 0, 1) * TranslateMatrix(vec3(RADIUS, RADIUS, 0.0f)) * rungAnimationRotationMatrix * TranslateMatrix(vec3(middle.x, middle.y, 0));
		

		for (int i = 0; i < RUNG_NUMBER; i++) {
			mat4 rotationMatrix = RotationMatrix(2.0f * 3.1415926f * (float)i / (float)RUNG_NUMBER, vec3(0, 0, 1));
			vec4 rotatedPoint = rungPoint * TranslateMatrix(vec3(-middle.x, -middle.y, 0)) * rotationMatrix * TranslateMatrix(vec3(middle.x, middle.y, 0));
			rungPoints.push_back(middle.x +perpendicular.x);
			rungPoints.push_back(middle.y + perpendicular.y);
			rungPoints.push_back(rotatedPoint.x + perpendicular.x);
			rungPoints.push_back(rotatedPoint.y + perpendicular.y);
		}

		glBufferData(GL_ARRAY_BUFFER, rungPoints.size() * sizeof(float), &rungPoints[0], GL_DYNAMIC_DRAW);
		glDrawArrays(GL_LINE_STRIP, 0, RUNG_NUMBER * 2);

		//draw head
		const float HEAD_DISTANCE = 2.1f;
		const float HEAD_RADIUS = 0.2f;
		vec4 headMiddle = middle * TranslateMatrix(vec3(0.0f, HEAD_DISTANCE, 0.0f));
		vec4 headPoint = vec4(headMiddle.x, headMiddle.y, 0, 1) * TranslateMatrix(vec3(HEAD_RADIUS, HEAD_RADIUS, 0.0f));
		
		headPoints.push_back(headPoint.x + perpendicular.x);
		headPoints.push_back(headPoint.y + perpendicular.y);
		for (int i = 0; i < TESSELATED_VERTICES + 1; i++) {
			mat4 rotationMatrix = RotationMatrix(2.0f * 3.1415926f * (float)i / (float)TESSELATED_VERTICES, vec3(0, 0, 1));
			vec4 rotatedPoint = headPoint * TranslateMatrix(vec3(-headMiddle.x, -headMiddle.y, 0)) * rotationMatrix * TranslateMatrix(vec3(headMiddle.x, headMiddle.y, 0));
			headPoints.push_back(rotatedPoint.x + perpendicular.x);
			headPoints.push_back(rotatedPoint.y + perpendicular.y);
		}

		glBufferData(GL_ARRAY_BUFFER, headPoints.size() * sizeof(float), &headPoints[0], GL_DYNAMIC_DRAW);
		glDrawArrays(GL_LINE_STRIP, 0, TESSELATED_VERTICES + 2);

		//draw body
		float bodyBottom = headMiddle.y + perpendicular.y - HEAD_RADIUS - HEAD_DISTANCE + RADIUS + RADIUS+0.4f;
		bodyPoints.push_back(headMiddle.x + perpendicular.x);
		bodyPoints.push_back(headMiddle.y + perpendicular.y - HEAD_RADIUS);
		bodyPoints.push_back(headMiddle.x + perpendicular.x);
		bodyPoints.push_back(bodyBottom);

		glBufferData(GL_ARRAY_BUFFER, bodyPoints.size() * sizeof(float), &bodyPoints[0], GL_DYNAMIC_DRAW);
		glDrawArrays(GL_LINE_STRIP, 0, 2);

		const float LEG_LENGTH = 0.8f;
		//Draw leg 1

		std::vector<float> kneePoints = circle_circle_intersection(middle.x+ perpendicular.x, bodyBottom, LEG_LENGTH, rungPoints[10], rungPoints[11], LEG_LENGTH);
		leftLegPoints.push_back(middle.x + perpendicular.x);
		leftLegPoints.push_back(bodyBottom);
		if (direction_right) {
			leftLegPoints.push_back(kneePoints[0] );
			leftLegPoints.push_back(kneePoints[1]);
		}
		else {
			leftLegPoints.push_back(kneePoints[2] );
			leftLegPoints.push_back(kneePoints[3]);
		}

		leftLegPoints.push_back(rungPoints[10] );
		leftLegPoints.push_back(rungPoints[11] );

		glBufferData(GL_ARRAY_BUFFER, leftLegPoints.size() * sizeof(float), &leftLegPoints[0], GL_DYNAMIC_DRAW);
		glDrawArrays(GL_LINE_STRIP, 0, 3);

		//Draw leg 2
		kneePoints = circle_circle_intersection(middle.x+ perpendicular.x, bodyBottom, LEG_LENGTH, rungPoints[rungPoints.size() - 2] , rungPoints[rungPoints.size() - 1] , LEG_LENGTH);
		rightLegPoints.push_back(middle.x + perpendicular.x);
		rightLegPoints.push_back(bodyBottom);
		if (direction_right) {
			rightLegPoints.push_back(kneePoints[0]);
			rightLegPoints.push_back(kneePoints[1]);
		}
		else {
			rightLegPoints.push_back(kneePoints[2]);
			rightLegPoints.push_back(kneePoints[3]);
		}


		rightLegPoints.push_back(rungPoints[rungPoints.size()-2]);
		rightLegPoints.push_back(rungPoints[rungPoints.size()-1] );

		glBufferData(GL_ARRAY_BUFFER, rightLegPoints.size() * sizeof(float), &rightLegPoints[0], GL_DYNAMIC_DRAW);
		glDrawArrays(GL_LINE_STRIP, 0, 3);

	
	}

	virtual float tStart() { return 0; }
	virtual float tEnd() { return 1; }

	virtual void AddControlPoint(float cX, float cY) {
		middle = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
	}


};
Circle* circle;





class KochanekBartelsCurve : public Curve {
public:
	std::vector<float> ts;
	float tension;
	void AddControlPoint(float cX, float cY) {
		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		printf("Adding control point at: %f %f", wVertex.x, wVertex.y);
		if (wCtrlPoints.size() < 4) {
			wCtrlPoints.push_back(wVertex);
			ts.push_back(wVertex.x);
		}
		else {
			for (int i = 0; i < wCtrlPoints.size() - 1; i++) {
				if (wCtrlPoints[i].x < wVertex.x && wCtrlPoints[i + 1].x > wVertex.x) {
					wCtrlPoints.insert(wCtrlPoints.begin() + i + 1, wVertex);
					ts.insert(ts.begin() + i +1, wVertex.x);
					break;
				}
			}
		}
		printf("%d size\n", ts.size());

	}

	KochanekBartelsCurve(float tension) {
		this->tension = tension;
	}

	void translateAllControlPoints(vec4 translateVector) {
		for (int i = 0; i < wCtrlPoints.size(); i++) {
			wCtrlPoints[i] = wCtrlPoints[i] * TranslateMatrix(vec3(translateVector.x, translateVector.y, 0.0f));
			ts[i] = wCtrlPoints[i].x;
		}
	}

	void setTension(float tension) {
		this->tension = tension;
	}

	float getTension() {
		return this->tension;
	}


	virtual void setFill(float point1, float point2, float point3, float point4) {
		fill1 = point1;
		fill2 = point2;
		fill3 = point3;
		fill4 = point4;
	}



	vec4 hermite_interpolation(vec4 p0, vec4 v0, float t0, vec4 p1, vec4 v1, float t1, float t) {
		vec4 a0 = p0;
		vec4 a1 = v0;
		vec4 a2 = ((p1 - p0) * 3.0 / pow((t1 - t0), 2)) - (v1 + v0 * 2.0) / (t1 - t0);
		vec4 a3 = (p0 - p1) * 2.0 / pow(t1 - t0, 3) + ((v1 + v0) / pow(t1 - t0, 2));
		return a3 * pow(t - t0, 3) + a2 * pow(t - t0, 2) + a1 * (t - t0) + a0;
	}

	float getY(float x) {
		if (wCtrlPoints.size() >= 4) {
			for (int i = 0; i < wCtrlPoints.size() - 2; i++) {
				if (ts[i] <= x && x <= ts[i + 1]) {
					vec4 v0;
					vec4 v1;
					if (i == 0) {
						v0 = (0.0f, 0.0f, 0.0f, 0.0f);
					}
					else {
						v0 = ((wCtrlPoints[i + 1] - wCtrlPoints[i]) / (ts[i + 1] - ts[i]) + (wCtrlPoints[i] - wCtrlPoints[i - 1]) / (ts[i] - ts[i - 1])) * ((1 - tension) / 2);
					}
					v1 = ((wCtrlPoints[i + 2] - wCtrlPoints[i + 1]) / (ts[i + 2] - ts[i + 1]) + (wCtrlPoints[i + 1] - wCtrlPoints[i]) / (ts[i + 1] - ts[i])) * ((1 - tension) / 2);
					return hermite_interpolation(wCtrlPoints[i], v0, ts[i], wCtrlPoints[i + 1], v1, ts[i + 1], x).y;
				}
			}
		}
	}



private:
	virtual vec4 r(float t) {
		if (wCtrlPoints.size() >= 4) {
			const float tension = -0.8f;
			for (int i = 0; i < wCtrlPoints.size() - 2; i++) {
				if (ts[i] <= t && t <= ts[i + 1]) {
					vec4 v0;
					vec4 v1;
					if (i == 0) {
						v0 = (0.0f, 0.0f, 0.0f, 0.0f);
					}
					else {
						v0 = ((wCtrlPoints[i + 1] - wCtrlPoints[i]) / (ts[i + 1] - ts[i]) + (wCtrlPoints[i] - wCtrlPoints[i - 1]) / (ts[i] - ts[i - 1])) * ((1 - tension)/2);
					}
					v1 = ((wCtrlPoints[i + 2] - wCtrlPoints[i + 1]) / (ts[i + 2] - ts[i + 1]) + (wCtrlPoints[i + 1] - wCtrlPoints[i]) / (ts[i + 1] - ts[i])) * ((1 - tension)/2);
					return hermite_interpolation(wCtrlPoints[i], v0, ts[i], wCtrlPoints[i + 1], v1, ts[i + 1], t);
				}
			}
		}
	}

};



bool followCircle = false;
TexturedQuad txq = TexturedQuad();

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	txq.Create();
	skyCurve = new KochanekBartelsCurve(-0.5f);
	skyCurve->AddControlPoint(-1.0f, 1.0f);
	skyCurve->AddControlPoint(0.0f, 0.8f);
	skyCurve->AddControlPoint(0.5f, 0.5f);
	skyCurve->AddControlPoint(1.0f, 1.0f);
	skyCurve->AddControlPoint(.99f, 1.0f);
	skyCurve->setFill(-50.0f, 50.0f, 50.0f, 50.0f);
	skyCurve->setColor(vec4(0.4f, 0.8f, 1.0f, 1.0f));

	curve = new KochanekBartelsCurve(0.5f);
	curve->setFill(-50.0f, -50.0f, 50.0f, -50.f);
	curve->AddControlPoint(-1.0f, -0.5f);
	curve->AddControlPoint(0.0f, -0.5f);
	curve->AddControlPoint(0.5f, -0.5f);
	curve->AddControlPoint(1.0f, -0.5f);
	curve->AddControlPoint(0.99f, -0.5f);
	curve->setColor(vec4(0.376f, 0.502f, 0.22f, 1.0f));

	circle = new Circle();
	circle->AddControlPoint(-0.8f, 0.3f);
	gpuProgram.Create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer
	txq.Draw();
	skyCurve->Draw();
	curve->Draw();
	if (curve->getPoints() >= 4) {
		circle->Draw();
	}
	// Set color to (0, 1, 0) = green
	int location = glGetUniformLocation(gpuProgram.getId(), "color");
	glUniform3f(location, 0.0f, 1.0f, 0.0f); // 3 floats

	float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
		0, 1, 0, 0,    // row-major!
		0, 0, 1, 0,
		0, 0, 0, 1 };

	location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
	glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location

	glBindVertexArray(vao);  // Draw call
	glDrawArrays(GL_LINE_STRIP, 0 /*startIdx*/, 3 /*# Elements*/);
	if (followCircle) {
		vec4 circleCenter = vec4(circle->getCenter().x, circle->getCenter().y, 0.0f, 1.0f);
		vec4 distance = camera.getCenter() - circleCenter;
		camera.setCenter(circle->getCenter());
		skyCurve->translateAllControlPoints(distance * -1.0f);
		
	}
	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') {
		glutPostRedisplay();
	} // if d, invalidate display, i.e. redraw
	if (key == 'q') {

	}
	if (key == 'o') {
		curve->setTension(curve->getTension() + 0.1f);
		printf("%f \n", curve->getTension());

	}
	if (key == 'l') {
		curve->setTension(curve->getTension() - 0.1f);
		printf("%f \n", curve->getTension());



	}
	if (key == ' ') {
		followCircle = !followCircle;
	}
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
										// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
													  // Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	char * buttonStat;
	if (state == GLUT_DOWN && button == GLUT_LEFT_BUTTON) {
		curve->AddControlPoint(cX, cY);

	}
	else if (state == GLUT_DOWN && button == GLUT_RIGHT_BUTTON) {
		circle->AddControlPoint(cX, cY);
	}
	switch (state) {
	case GLUT_DOWN:
		buttonStat = "pressed";
		break;
	case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:
		printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);
		break;
	case GLUT_MIDDLE_BUTTON:
		printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);
		break;
	case GLUT_RIGHT_BUTTON:
		printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);
		break;
	}
}



// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	tCurrent = time / 1000.0f;
	glutPostRedisplay();

}