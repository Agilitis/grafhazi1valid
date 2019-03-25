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

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(0, 1, 0, 1);	// computed color is the color of the primitive
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
};

Camera2D camera;	// 2D camera
bool animate = true;
float tCurrent = 0;	// current clock in sec
const int nTesselatedVertices = 2000;
const vec4 gravity = vec4(0.0f, -0.001f, 0.0f, 1.0f);

class Curve {
	unsigned int vaoCurve, vboCurve;
	unsigned int vaoCtrlPoints, vboCtrlPoints;
	unsigned int vaoAnimatedObject, vboAnimatedObject;
	int currentSize;
	std::vector<vec4> linePoints;
protected:
	std::vector<vec4> wCtrlPoints;		// coordinates of control points
public:
	int getPoints() {
		return wCtrlPoints.size();
	}

	Curve() {
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

	virtual vec4 getPointAtPosition(float position) {
		return 0;
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



		if (wCtrlPoints.size() > 0) {	// draw control points
			glBindVertexArray(vaoCtrlPoints);
			glBindBuffer(GL_ARRAY_BUFFER, vboCtrlPoints);
			glBufferData(GL_ARRAY_BUFFER, wCtrlPoints.size() * 4 * sizeof(float), &wCtrlPoints[0], GL_DYNAMIC_DRAW);
			if (colorLocation >= 0) glUniform3f(colorLocation, 1, 0, 0);
			glPointSize(1.0f);
			glDrawArrays(GL_POINTS, 0, wCtrlPoints.size());
		}

		if (wCtrlPoints.size() >= 4) {	// draw curve
			std::vector<float> vertexData;
			for (int i = 0; i < nTesselatedVertices; i++) {	// Tessellate
				float tNormalized = (float)i / (nTesselatedVertices - 1);
				float t = tStart() + (tEnd() - tStart()) * tNormalized;
				vec4 wVertex = r(t);
				vertexData.push_back(wVertex.x);
				vertexData.push_back(wVertex.y);
			}
			// copy data to the GPU
			glBindVertexArray(vaoCurve);
			glBindBuffer(GL_ARRAY_BUFFER, vboCurve);
			glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), &vertexData[0], GL_DYNAMIC_DRAW);
			if (colorLocation >= 0) glUniform3f(colorLocation, 1, 0.5f, 0);
			glDrawArrays(GL_LINE_STRIP, 0, nTesselatedVertices);

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
};



Curve * curve;

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
		std::vector<float> middlePoint;
		std::vector<float> circlePoints;
		std::vector<float> rungPoints;
		middlePoint.push_back(middle.x);
		middlePoint.push_back(middle.y);
		float rotationSpeed = tCurrent / -10.0f;
		//setup rotation matrix for animation	
		if (animate) {
			std::vector<vec4> linePoints = curve->getLinePoints();
			movementVector += gravity;

			for (int i = 0; i < nTesselatedVertices - 1; i++) {
				//Check if X coordiante is good
				if (middle.x > linePoints[i].x && middle.x < linePoints[i + 1].x) {
					//on curve
					if (middle.y-(RADIUS*2) <= linePoints[i].y) {
						movementVector = linePoints[i + 1] - linePoints[i];
					}
					//above curve
					else {

					}
				}
			}
			

			//Rotating the circle
			rungAnimationRotationMatrix = RotationMatrix(2.0f * 3.1415926f * rotationSpeed, vec3(0, 0, 1));

			//Moving the circle
			middle = middle * TranslateMatrix(vec3(movementVector.x, movementVector.y, movementVector.z));

			
		}


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
		circlePoints.push_back(circlePoint.x);
		circlePoints.push_back(circlePoint.y);
		for (int i = 0; i < TESSELATED_VERTICES+1; i++) {
			mat4 rotationMatrix = RotationMatrix(2.0f * 3.1415926f * (float)i / (float)TESSELATED_VERTICES, vec3(0,0,1));
			vec4 rotatedPoint = circlePoint * TranslateMatrix(vec3(-middle.x, -middle.y,0)) * rotationMatrix * TranslateMatrix(vec3(middle.x, middle.y, 0));
			circlePoints.push_back(rotatedPoint.x);
			circlePoints.push_back(rotatedPoint.y);
		}

		glBufferData(GL_ARRAY_BUFFER, circlePoints.size() * sizeof(float), &circlePoints[0], GL_DYNAMIC_DRAW);
		glDrawArrays(GL_LINE_STRIP, 0, TESSELATED_VERTICES+2);

		//draw rungs
		const int RUNG_NUMBER = 6;
		vec4 rungPoint;
		if (animate) {
			rungPoint = vec4(0, 0, 0, 1) * TranslateMatrix(vec3(RADIUS, RADIUS, 0.0f)) * rungAnimationRotationMatrix * TranslateMatrix(vec3(middle.x, middle.y, 0));
		}
		else {
			rungPoint = vec4(middle.x, middle.y, 0, 1) * TranslateMatrix(vec3(RADIUS, RADIUS, 0.0f));
		}


		for (int i = 0; i < RUNG_NUMBER; i++) {
			mat4 rotationMatrix = RotationMatrix(2.0f * 3.1415926f * (float)i / (float)RUNG_NUMBER, vec3(0, 0, 1));
			vec4 rotatedPoint = rungPoint * TranslateMatrix(vec3(-middle.x, -middle.y, 0)) * rotationMatrix * TranslateMatrix(vec3(middle.x, middle.y, 0));
			rungPoints.push_back(middle.x);
			rungPoints.push_back(middle.y);
			rungPoints.push_back(rotatedPoint.x);
			rungPoints.push_back(rotatedPoint.y);
		}

		glBufferData(GL_ARRAY_BUFFER, rungPoints.size() * sizeof(float), &rungPoints[0], GL_DYNAMIC_DRAW);
		glDrawArrays(GL_LINE_STRIP, 0, RUNG_NUMBER*2);

	
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
	void AddControlPoint(float cX, float cY) {
		printf("Adding control point!");
		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		ts.push_back(wCtrlPoints.size()*0.01f);
		wCtrlPoints.push_back(wVertex);
		for (int i = 0; i < ts.size(); i++) {
			ts[i] = (1.0f / (ts.size() + 1)) * i;
			printf("%f \n", ts[i]);
		}
		printf("%d size\n", ts.size());

	}
	vec4 hermite_interpolation(vec4 p0, vec4 v0, float t0, vec4 p1, vec4 v1, float t1, float t) {
		vec4 a0 = p0;
		vec4 a1 = v0;
		vec4 a2 = ((p1 - p0) * 3.0 / pow((t1 - t0), 2)) - (v1 + v0 * 2.0) / (t1 - t0);
		vec4 a3 = (p0 - p1) * 2.0 / pow(t1 - t0, 3) + ((v1 + v0) / pow(t1 - t0, 2));
		return a3 * pow(t - t0, 3) + a2 * pow(t - t0, 2) + a1 * (t - t0) + a0;
	}



private:
	virtual vec4 r(float t) {
		if (wCtrlPoints.size() >= 4) {
			const float tension = 0.5f;
			for (int i = 0; i < wCtrlPoints.size()-2; i++) {
				if (ts[i] <= t && t <= ts[i + 1]) {
					vec4 v0;
					vec4 v1;
					if (i == 0) {
						v0 = (0.0f, 0.0f, 0.0f, 0.0f);
					}
					else {
						v0 = ((wCtrlPoints[i + 1] - wCtrlPoints[i]) / (ts[i + 1] - ts[i]) + (wCtrlPoints[i] - wCtrlPoints[i - 1]) / (ts[i] - ts[i - 1])) * (1-tension);
					}
					v1 = ((wCtrlPoints[i + 2] - wCtrlPoints[i + 1]) / (ts[i + 2] - ts[i + 1]) + (wCtrlPoints[i + 1] - wCtrlPoints[i]) / (ts[i + 1] - ts[i])) * (1-tension);
					return hermite_interpolation(wCtrlPoints[i], v0, ts[i], wCtrlPoints[i + 1], v1, ts[i + 1], t);
				}
			}
		}
	}

};






void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	curve = new KochanekBartelsCurve();
	curve->AddControlPoint(-1.0f, 0.0f);
	circle = new Circle();
	circle->AddControlPoint(-0.8f, 0.3f);
	gpuProgram.Create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer
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

	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
	if (key == 'q') {
		circle->AddControlPoint(1.0f, 0.0f);

	}
	if (key == 'l') {
		camera.setCenter(circle->getCenter());
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