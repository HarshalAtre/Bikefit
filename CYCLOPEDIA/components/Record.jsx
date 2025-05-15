import React, { useEffect, useState, useRef } from 'react';
import { StyleSheet, View, Text, TouchableOpacity, ActivityIndicator, Alert } from 'react-native';
import { Camera, useCameraDevice, useCameraPermission } from 'react-native-vision-camera';
import { launchImageLibrary } from 'react-native-image-picker';
import { useNavigation, useRoute } from '@react-navigation/native';
import Video from 'react-native-video';

const Record = () => {
  const device = useCameraDevice('back');
  const { hasPermission, requestPermission } = useCameraPermission();
  const camera = useRef(null);
  const [isRecording, setIsRecording] = useState(false);

  const [rearVideoUri, setRearVideoUri] = useState(null);
  const [sideVideoUri, setSideVideoUri] = useState(null);

  const [previewUri, setPreviewUri] = useState(null);
  const [currentVideoType, setCurrentVideoType] = useState('rear'); // 'rear' or 'side'
  const [isUploading, setIsUploading] = useState(false);

  const navigation = useNavigation();
  const route = useRoute();
  const { inputData, photos } = route.params;

  useEffect(() => {
    requestPermission();
  }, []);

  if (!hasPermission) return <Text>Requesting camera permission...</Text>;
  if (!device) return <Text>No Camera Device Found</Text>;

  const startRecording = async () => {
    if (camera.current) {
      setIsRecording(true);
      try {
        await camera.current.startRecording({
          fileType: 'mp4',
          onRecordingFinished: (video) => {
            setPreviewUri(`file://${video.path}`);
            setIsRecording(false);
          },
          onRecordingError: (error) => {
            console.error('Recording Error:', error);
            setIsRecording(false);
          },
        });
      } catch (error) {
        console.error('Start Recording Error:', error);
        setIsRecording(false);
      }
    }
  };

  const stopRecording = async () => {
    if (camera.current) {
      camera.current.stopRecording();
    }
  };

  const pickVideo = () => {
    launchImageLibrary({ mediaType: 'video' }, (response) => {
      if (response.didCancel) return;
      if (response.errorCode) {
        Alert.alert('Error', 'Failed to pick the video.');
        return;
      }
      setPreviewUri(response.assets[0].uri);
    });
  };

  const confirmPreview = () => {
    if (currentVideoType === 'rear') {
      setRearVideoUri(previewUri);
      setCurrentVideoType('side');
    } else {
      setSideVideoUri(previewUri);
    }
    setPreviewUri(null);
  };

  const editVideo = (type) => {
    setCurrentVideoType(type);
    if (type === 'rear') {
      setRearVideoUri(null);
    } else {
      setSideVideoUri(null);
    }
  };
  const uploadVideos = async () => {
    if (!rearVideoUri || !sideVideoUri) return;
  
    const formData = new FormData();
  
    formData.append("height", inputData.height);
    formData.append("inseamLength", inputData.inseamLength);
    formData.append("armLength", inputData.armLength);
  
    photos.forEach((uri, index) => {
      formData.append('photos', {
        uri,
        type: 'image/jpeg',
        name: `photo_${index}.jpg`,
      });
    });
  
    formData.append('rearVideo', {
      uri: rearVideoUri,
      name: 'rear_video.mp4',
      type: 'video/mp4',
    });
  
    formData.append('sideVideo', {
      uri: sideVideoUri,
      name: 'side_video.mp4',
      type: 'video/mp4',
    });
  
    setIsUploading(true);
    navigation.replace('Loading');
  
    try {
      // const res = await fetch('http://192.168.29.130:5000/upload', {
      const res = await fetch('https://harshal3304-bike-fit.hf.space/upload', {

        method: 'POST',
        body: formData,
        headers: {
          // Let fetch set proper Content-Type automatically
          // DON'T add 'Content-Type': 'multipart/form-data' manually
        },
      });
  
      if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
      const json = await res.json();
      console.log("Received JSON from server:", json);
      navigation.replace('Result', { predictions: json });
    } catch (error) {
      console.error('Upload Error:', error);
      Alert.alert('Upload Failed', error.message);
      navigation.goBack();
    } finally {
      setIsUploading(false);
    }
  };
  


  // Rendering Logic
  if (previewUri) {
    return (
      <View style={styles.previewContainer}>
        <Video source={{ uri: previewUri }} controls style={styles.videoPreview} resizeMode="contain" repeat />
        <View style={styles.controls}>
          <TouchableOpacity onPress={() => setPreviewUri(null)} style={styles.buttonCancel}>
            <Text style={styles.buttonText}>Retake</Text>
          </TouchableOpacity>
          <TouchableOpacity onPress={confirmPreview} style={styles.buttonConfirm}>
            <Text style={styles.buttonText}>Confirm</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {(!rearVideoUri || !sideVideoUri) && (
        <Camera ref={camera} style={StyleSheet.absoluteFill} device={device} isActive video />
      )}

      <View style={styles.controls}>
        {isRecording ? (
         <TouchableOpacity onPress={stopRecording} style={styles.buttonStop}>
            <Text style={styles.buttonText}>Stop</Text>
          </TouchableOpacity>
        ) : (
           <>
            {!isUploading && <TouchableOpacity onPress={startRecording} style={styles.button}>
              <Text style={styles.buttonText}>Record {currentVideoType} View</Text>
            </TouchableOpacity>}
            <TouchableOpacity onPress={pickVideo} style={styles.buttonUpload}>
              <Text style={styles.buttonText}>Upload {currentVideoType} View</Text>
            </TouchableOpacity>
          </>
        )}
      </View>

      {rearVideoUri && sideVideoUri && (
        <View style={styles.finalPreview}>
          <View style={styles.videoBlock}>
            <Text style={styles.label}>Rear View</Text>
            <Video source={{ uri: rearVideoUri }} style={styles.previewThumb} resizeMode="cover" repeat />
            <TouchableOpacity onPress={() => editVideo('rear')} style={styles.buttonEdit}>
              <Text style={styles.buttonText}>Edit Rear</Text>
            </TouchableOpacity>
          </View>

          <View style={styles.videoBlock}>
            <Text style={styles.label}>Side View</Text>
            <Video source={{ uri: sideVideoUri }} style={styles.previewThumb} resizeMode="cover" repeat />
            <TouchableOpacity onPress={() => editVideo('side')} style={styles.buttonEdit}>
              <Text style={styles.buttonText}>Edit Side</Text>
            </TouchableOpacity>
          </View>

          <TouchableOpacity onPress={uploadVideos} style={styles.buttonProcess}>
            {isUploading ? <ActivityIndicator color="white" /> : <Text style={styles.buttonText}>Upload & Process</Text>}
          </TouchableOpacity>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1 },
  previewContainer: { flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'black',},
  videoPreview: { width: '100%',
    height: '70%',},
  controls: { position: 'absolute',
    bottom: 50,
    alignSelf: 'center',
    flexDirection: 'row',
    gap: 10, },
  button: { backgroundColor: 'green', padding: 10, borderRadius: 5 },
  buttonUpload: { backgroundColor: 'orange', padding: 10, borderRadius: 5 },
  buttonStop: { backgroundColor: 'red', padding: 10, borderRadius: 5 },
  buttonConfirm: { backgroundColor: 'blue', padding: 10, borderRadius: 5 },
  buttonCancel: { backgroundColor: 'gray', padding: 10, borderRadius: 5 },
  buttonEdit: { backgroundColor: 'purple', padding: 8, borderRadius: 5, marginTop: 5 },
  buttonProcess: {
    backgroundColor: 'blue',
    padding: 12,
    borderRadius: 5,
    alignSelf: 'center',
    marginTop: 50,
  },
  buttonText: { color: 'white', fontWeight: 'bold' },
  finalPreview: { marginTop: 130, alignItems: 'center' },
  videoBlock: { marginBottom: 15, alignItems: 'center' },
  previewThumb: { width: 200, height: 150, borderRadius: 10 },
  label: { fontWeight: 'bold', marginBottom: 5, color: '#333' },
});

export default Record;
