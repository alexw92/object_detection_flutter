import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:google_ml_kit/google_ml_kit.dart';
import 'package:flutter_tflite/flutter_tflite.dart';


void main() => runApp(ObjectDetectionApp());

class ObjectDetectionApp extends StatefulWidget {
  @override
  _ObjectDetectionAppState createState() => _ObjectDetectionAppState();
}

class VisionObject {
  final String label;
  final double confidence;
  final Rect boundingBox;

  VisionObject({
    this.label = '',
    this.confidence = 0.0,
    this.boundingBox = const Rect.fromLTRB(0, 0, 0, 0),
  }) ;
}

class _ObjectDetectionAppState extends State<ObjectDetectionApp> {
  late CameraController _cameraController;
  late List<VisionObject> _visionObjects;
  bool _isDetecting = false;
  String _model = "YOUR_CUSTOM_MODEL_NAME";

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  void _initializeCamera() async {
    final cameras = await availableCameras();
    final firstCamera = cameras.first;
    _cameraController = CameraController(
      firstCamera,
      ResolutionPreset.medium,
    );
    _cameraController.initialize().then((_) {
      if (!mounted) {
        return;
      }
      setState(() {});

      _cameraController.startImageStream((CameraImage image) {
        if (_isDetecting) return;

        _isDetecting = true;

        _detectObjects(image).then((dynamic result) {
          setState(() {
            _visionObjects = result;
          });

          _isDetecting = false;
        });
      });
    });
  }

  void _detectObjects() async {
    if (_cameraController.value.isInitialized && !_isDetecting) {
      setState(() {
        _isDetecting = true;
      });

      final image = _cameraController.value.previewSize;
      final input = TensorImage.fromBytes(
        image.height,
        image.width,
        image.data,
        mean: 127.5,
        std: 127.5,
      );
      final output = await Tflite.runModelOnFrame(
        image: input,
        imageMean: 0.0,
        imageStd: 255.0,
      );
      final List<VisionObject> visionObjects = _processOutput(output);

      setState(() {
        _visionObjects = visionObjects;
        _isDetecting = false;
      });
    }
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        body: _cameraController == null
            ? Center(child: CircularProgressIndicator())
            : Stack(
          children: <Widget>[
            CameraPreview(_cameraController),
            _buildObjectDetectionOverlay(context),
          ],
        ),
      ),
    );
  }
}
