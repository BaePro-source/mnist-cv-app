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

  Future<void> _predictImage() async {
    if (_imageFile == null) {
      setState(() {
        _result = '먼저 사진을 선택해주세요';
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
                    ? const Text('선택한 사진이 없습니다')
                    : Image.file(_imageFile!),
              ),
            ),
            const SizedBox(height: 16),
            Text(
              _result,
              style: const TextStyle(fontSize: 20),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: ElevatedButton(
                    onPressed: _takePhoto,
                    child: const Text('사진 찍기'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: ElevatedButton(
                    onPressed: _pickFromGallery,
                    child: const Text('사진첩에서 가져오기'),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: _isLoading ? null : _predictImage,
                child: _isLoading
                    ? const SizedBox(
                        width: 24,
                        height: 24,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Text('예측하기'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}