import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ActivityIndicator, Alert } from 'react-native';
import { launchImageLibrary } from 'react-native-image-picker';
import RNFS from 'react-native-fs';
import { useNavigation, useRoute } from '@react-navigation/native';

const Upload = () => {
  const [rearVideoUri, setRearVideoUri] = useState(null);
  const [sideVideoUri, setSideVideoUri] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const navigation = useNavigation();
  const route = useRoute();
  const { height, inseamLength, armLength } = route.params;

  const pickVideo = (type) => {
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
        if (type === 'rear') {
          setRearVideoUri(uri);
        } else {
          setSideVideoUri(uri);
        }
      }
    });
  };

  const uploadVideos = async () => {
    if (!rearVideoUri || !sideVideoUri) return;

    setIsUploading(true);

    // Navigate to Loading Screen
    navigation.replace('Loading');

    const formData = new FormData();
    formData.append('height', height);
    formData.append('inseamLength', inseamLength);
    formData.append('armLength', armLength);
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
    console.log(formData)
    
    try {
      const response = await fetch('http://192.168.29.130:5000/process_videos', {
        method: 'POST',
        body: formData,
      });

      const jsonResponse = await response.json();
      console.log("prediction: ", jsonResponse);

      // Navigate to Result Screen with the predictions
      navigation.replace('Result', {
        predictions: jsonResponse.predictions,
      });
    } catch (error) {
      console.error('Upload Error:', error);
      setIsUploading(false);
      navigation.navigate('Home');
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.text}>Upload Screen</Text>

      <TouchableOpacity onPress={() => pickVideo('rear')} style={styles.button}>
        <Text style={styles.buttonText}>Pick Rear View Video</Text>
      </TouchableOpacity>

      <TouchableOpacity onPress={() => pickVideo('side')} style={styles.button}>
        <Text style={styles.buttonText}>Pick Side View Video</Text>
      </TouchableOpacity>

      {rearVideoUri && sideVideoUri && (
        <TouchableOpacity onPress={uploadVideos} style={styles.buttonUpload}>
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
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f5f5f5',
  },
  text: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  button: {
    backgroundColor: 'green',
    padding: 15,
    borderRadius: 10,
    marginVertical: 10,
    width: '80%',
    alignItems: 'center',
  },
  buttonUpload: {
    backgroundColor: 'blue',
    padding: 15,
    borderRadius: 10,
    marginVertical: 10,
    width: '80%',
    alignItems: 'center',
  },
  buttonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
});

export default Upload;