import React, { useEffect, useState, useRef } from 'react';
import { StyleSheet, View, Text, TouchableOpacity, ActivityIndicator, Alert } from 'react-native';
import { Camera, useCameraDevice, useCameraPermission } from 'react-native-vision-camera';
import { launchImageLibrary } from 'react-native-image-picker';
import RNFS from 'react-native-fs';
import { useNavigation, useRoute } from '@react-navigation/native';

const Record = () => {
  const device = useCameraDevice('back');
  const { hasPermission, requestPermission } = useCameraPermission();
  const camera = useRef(null);
  const [isRecording, setIsRecording] = useState(false);
  const [rearVideoUri, setRearVideoUri] = useState(null);
  const [sideVideoUri, setSideVideoUri] = useState(null);
  const [currentVideoType, setCurrentVideoType] = useState('rear'); // 'rear' or 'side'
  const [isUploading, setIsUploading] = useState(false);
  const navigation = useNavigation();
  const route = useRoute();
  const { height, inseamLength, armLength } = route.params;

  useEffect(() => {
    requestPermission();
  }, []);

  if (!hasPermission) return <Text>Requesting camera permission...</Text>;
  if (device == null) return <Text>No Camera Device Found</Text>;

  const startRecording = async () => {
    if (camera.current) {
      try {
        setIsRecording(true);
        console.log('Starting recording...');
        await camera.current.startRecording({
          fileType: 'mp4',
          onRecordingFinished: (video) => {
            console.log('Recorded Video Path:', video.path);
            console.log('Video URI:', `file://${video.path}`);
            if (currentVideoType === 'rear') {
              setRearVideoUri(video.path);
              setCurrentVideoType('side'); // Switch to Side View after recording
            } else {
              setSideVideoUri(video.path);
            }
            setIsRecording(false);
          },
          onRecordingError: (error) => {
            console.error('Recording Error:', error);
            setIsRecording(false);
          },
        });
      } catch (error) {
        console.error('Start Recording Error:', error);
      }
    }
  };

  const stopRecording = async () => {
    if (camera.current) {
      camera.current.stopRecording();
    }
  };

  const pickVideo = async () => {
    const options = {
      mediaType: 'video',
      includeBase64: false,
    };

    launchImageLibrary(options, (response) => {
      if (response.didCancel) {
        console.log('User cancelled video picker');
      } else if (response.error) {
        console.log('ImagePicker Error: ', response.error);
        Alert.alert('Error', 'Failed to pick the video.');
      } else if (response.assets) {
        const uri = response.assets[0].uri;
        if (currentVideoType === 'rear') {
          setRearVideoUri(uri);
          setCurrentVideoType('side'); // Switch to Side View after uploading
        } else {
          setSideVideoUri(uri);
        }
      }
    });
  };

  const uploadVideos = async () => {
    if (!rearVideoUri || !sideVideoUri) return;

    // Add 'file://' prefix to URIs
    const rearVideoPath = `file://${rearVideoUri}`;
    const sideVideoPath = `file://${sideVideoUri}`;

    // Navigate to Loading Screen
    navigation.replace('Loading');

    const formData = new FormData();
    formData.append('height', height);
    formData.append('inseamLength', inseamLength);
    formData.append('armLength', armLength);
    formData.append('rearVideo', {
      uri: rearVideoPath,  // Use the corrected path
      name: 'rear_video.mp4',
      type: 'video/mp4',
    });
    formData.append('sideVideo', {
      uri: sideVideoPath,  // Use the corrected path
      name: 'side_video.mp4',
      type: 'video/mp4',
    });

    try {
      console.log('Uploading videos...');
      const response = await fetch('http://192.168.29.130:5000/process_videos', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Server responded with ${response.status}: ${errorText}`);
      }

      const jsonResponse = await response.json();
      console.log('Prediction: ', jsonResponse);

      navigation.replace('Result', {
        predictions: jsonResponse.predictions,
      });
    } catch (error) {
      console.error('Upload Error:', error);
      navigation.goBack();
    }
  };

  return (
    <View style={styles.container}>
      <Camera
        ref={camera}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        video={true}
      />
      <View style={styles.controls}>
        {/* Upload Button */}

{ !sideVideoUri && (
        <TouchableOpacity onPress={pickVideo} style={styles.buttonUpload}>
          <Text style={styles.buttonText}>Upload {currentVideoType === 'rear' ? 'Rear View' : 'Side View'}</Text>
        </TouchableOpacity>)
}


        {/* Record Button */}
        {!isRecording ? (
          !rearVideoUri || !sideVideoUri ? (
            <TouchableOpacity onPress={startRecording} style={styles.button}>
              <Text style={styles.buttonText}>
                Record {currentVideoType === 'rear' ? 'Rear View' : 'Side View'}
              </Text>
            </TouchableOpacity>
          ) : null
        ) : (
          <TouchableOpacity onPress={stopRecording} style={styles.buttonStop}>
            <Text style={styles.buttonText}>Stop Recording</Text>
          </TouchableOpacity>
        )}
      </View>

      {/* Upload & Process Button */}
      {rearVideoUri && sideVideoUri && (
        <TouchableOpacity onPress={uploadVideos} style={styles.buttonProcess}>
          {isUploading ? (
            <ActivityIndicator color="white" />
          ) : (
            <Text style={styles.buttonText}>Upload & Process</Text>
          )}
        </TouchableOpacity>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1 },
  controls: {
    position: 'absolute',
    bottom: 50,
    alignSelf: 'center',
    flexDirection: 'row',
    gap: 10,
  },
  button: {
    backgroundColor: 'green',
    padding: 10,
    borderRadius: 5,
  },
  buttonStop: {
    backgroundColor: 'red',
    padding: 10,
    borderRadius: 5,
  },
  buttonUpload: {
    backgroundColor: 'orange',
    padding: 10,
    borderRadius: 5,
  },
  buttonProcess: {
    backgroundColor: 'blue',
    padding: 10,
    borderRadius: 5,
    position: 'absolute',
    bottom: 120,
    alignSelf: 'center',
  },
  buttonText: {
    color: 'white',
    fontWeight: 'bold',
  },
});

export default Record;