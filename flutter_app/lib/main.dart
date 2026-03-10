import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final ImagePicker _picker = ImagePicker();

  File? _imageFile;
  String _result = '아직 예측 결과 없음';
  bool _isLoading = false;

  Future<void> _pickFromGallery() async {
  final XFile? pickedFile = await _picker.pickImage(
    source: ImageSource.gallery,
    imageQuality: 100,
  );

  if (pickedFile == null) return;

  setState(() {
    _imageFile = File(pickedFile.path);
    _result = '사진 선택 완료';
  });
}

  // Mac에서 실행 중인 FastAPI를 같은 와이파이의 폰으로 접속할 때
  // localhost 말고 맥의 실제 IP를 써야 함.
  final String serverUrl = 'http://192.168.0.63:8000/predict';

  Future<void> _takePhoto() async {
    final XFile? pickedFile = await _picker.pickImage(
      source: ImageSource.camera,
      imageQuality: 100,
    );

    if (pickedFile == null) return;

    setState(() {
      _imageFile = File(pickedFile.path);
      _result = '사진 선택 완료';
    });
  }

  Future<void> _predictImage() async {
    if (_imageFile == null) {
      setState(() {
        _result = '먼저 사진을 찍어주세요';
      });
      return;
    }

    setState(() {
      _isLoading = true;
      _result = '예측 중...';
    });

    try {
      final request = http.MultipartRequest('POST', Uri.parse(serverUrl));
      request.files.add(
        await http.MultipartFile.fromPath('file', _imageFile!.path),
      );

      final streamedResponse = await request.send();
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);

        setState(() {
          // 서버 응답 형식에 맞게 수정
          // 예: {"prediction": 7}
          _result = '예측 결과: ${data['prediction']}';
        });
      } else {
        setState(() {
          _result = '서버 오류: ${response.statusCode}\n${response.body}';
        });
      }
    } catch (e) {
      setState(() {
        _result = '요청 실패: $e';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('MNIST Camera App'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            Expanded(
              child: Center(
                child: _imageFile == null
                    ? const Text('촬영한 사진이 없습니다')
                    : Image.file(_imageFile!),
              ),
            ),
            const SizedBox(height: 16),
            Text(
              _result,
              style: const TextStyle(fontSize: 20),
            ),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: _takePhoto,
              child: const Text('사진 찍기'),
            ),
            const SizedBox(height: 12),
            ElevatedButton(
              onPressed: _isLoading ? null : _predictImage,
              child: _isLoading
                  ? const CircularProgressIndicator()
                  : const Text('예측하기'),
            ),
          ],
        ),
      ),
    );
  }
}