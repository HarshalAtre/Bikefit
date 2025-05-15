import React, { useEffect, useState, useRef } from 'react';
import { StyleSheet, View, Text, TouchableOpacity, Image, ScrollView, Alert,Modal  } from 'react-native';
import { Camera, useCameraDevice, useCameraPermission } from 'react-native-vision-camera';
import { launchImageLibrary } from 'react-native-image-picker';
import { useNavigation, useRoute } from '@react-navigation/native';

const PhotoCapture = () => {
  const device = useCameraDevice('back');
  const { hasPermission, requestPermission } = useCameraPermission();
  const camera = useRef(null);
  const [photos, setPhotos] = useState([]);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const navigation = useNavigation();
  const [previewVisible, setPreviewVisible] = useState(false);
  const [previewUri, setPreviewUri] = useState(null);
  const route = useRoute();
  const { height, inseamLength, armLength } = route.params;
  useEffect(() => {
    requestPermission();
  }, []);

  const takePhoto = async () => {
    if (camera.current && photos.length < 8) {
      try {
        const photo = await camera.current.takePhoto({
          flash: 'off',
        });
        setPhotos([...photos, `file://${photo.path}`]);
      } catch (error) {
        console.error('Error taking photo:', error);
        Alert.alert('Error', 'Failed to take photo.');
      }
    } else {            
      Alert.alert('Limit Reached', 'You can only add up to 8 photos.');
    }
  };

  const pickImage = () => {
    if (photos.length >= 8) {
      Alert.alert('Limit Reached', 'You can only add up to 8 photos.');
      return;
    }
  
    launchImageLibrary(
      {
        mediaType: 'photo',
        selectionLimit: 8 - photos.length, // allow only as many as needed to reach 8
      },
      (response) => {
        if (response.didCancel) {
          console.log('User cancelled image picker');
        } else if (response.errorCode) {
          console.log('ImagePicker Error: ', response.errorMessage);
          Alert.alert('Error', 'Failed to pick the image.');
        } else if (response.assets && response.assets.length > 0) {
          const uris = response.assets.map((asset) => asset.uri);
          setPhotos((prevPhotos) => [...prevPhotos, ...uris]);
        }
      }
    );
  };
  

  const removePhoto = (index) => {
    const updatedPhotos = [...photos];
    updatedPhotos.splice(index, 1);
    setPhotos(updatedPhotos);
  };

  const proceed = () => {
    if (photos.length !== 8) {
      Alert.alert('Invalid Photo Count', 'Please add exactly 8 photos to proceed.');
      return;
    }
  
    // Replace with your real form inputs
    const inputData = {
      height ,
      inseamLength,
      armLength
    };
  
    navigation.navigate('Record', {
      inputData,
      photos,
    });
  };

  if (!hasPermission) return <Text>Requesting camera permission...</Text>;
  if (device == null) return <Text>No Camera Device Found</Text>;

  return (
    <View style={styles.container}>
      {isCameraActive && (
        <Camera
          ref={camera}
          style={styles.camera}
          device={device}
          isActive={true}
          photo={true}
        />
      )}
      <View style={styles.controls}>

      {photos.length != 8 && <TouchableOpacity
  onPress={() => setIsCameraActive(!isCameraActive)}
  style={styles.button}
  disabled={photos.length >= 8}
>
  <Text style={styles.buttonText}>{isCameraActive ? 'Close Camera' : 'Open Camera'}</Text>
</TouchableOpacity>}

{photos.length != 8 && isCameraActive && (
  <TouchableOpacity
    onPress={takePhoto}
    style={styles.button}
    disabled={photos.length >= 8}
  >
    <Text style={styles.buttonText}>Take Photo</Text>
  </TouchableOpacity>
)}

{photos.length != 8 && <TouchableOpacity
  onPress={pickImage}
  style={styles.button}
  disabled={photos.length >= 8}
>
  <Text style={styles.buttonText}>Upload Photo</Text>
</TouchableOpacity>}

      </View>
      <ScrollView contentContainerStyle={styles.photoContainer}>
      {photos.map((uri, index) => (
  <TouchableOpacity key={index} onPress={() => {
    setPreviewUri(uri);
    setPreviewVisible(true);
  }}>
    <View style={styles.photoWrapper}>
      <Image source={{ uri }} style={styles.photo} />
      <TouchableOpacity onPress={() => removePhoto(index)} style={styles.removeButton}>
        <Text style={styles.removeButtonText}>X</Text>
      </TouchableOpacity>
    </View>
  </TouchableOpacity>
))}
      </ScrollView>

     {photos.length == 8 && <TouchableOpacity onPress={proceed} style={styles.proceedButton}>
        <Text style={styles.buttonText}>Proceed</Text>
      </TouchableOpacity>}
      {previewUri && (
  <Modal visible={previewVisible} transparent={true}>
    <View style={styles.previewContainer}>
      <TouchableOpacity style={styles.closePreview} onPress={() => setPreviewVisible(false)}>
        <Text style={styles.closeText}> X </Text>
      </TouchableOpacity>
      <Image source={{ uri: previewUri }} style={styles.fullImage} resizeMode="contain" />
    </View>
  </Modal>
)}
    </View>
    
  );
};

const styles = StyleSheet.create({
  container: { flex: 1 },
  camera: {
    flex: 50,
    width: '100%',
  height: 800, // increased height for better visibility
  },
  controls: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    padding: 10,
  },
  button: {
    backgroundColor: '#1E90FF',
    padding: 10,
    borderRadius: 5,
  },
  buttonText: {
    color: '#fff',
    fontWeight: 'bold',
  },
  photoContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    padding: 20,
    gap:5
  },
  photoWrapper: {
    position: 'relative',
    margin: 5,
  },
  photo: {
    width: 100,
    height: 100,
    borderRadius: 5,
  },
  removeButton: {
    position: 'absolute',
    top: 2,
    right: 2,
    backgroundColor: 'rgba(255,0,0,0.7)',
    borderRadius: 12,
    width: 24,
    height: 24,
    alignItems: 'center',
    justifyContent: 'center',
  },
  removeButtonText: {
    color: '#fff',
    fontWeight: 'bold',
  },
  proceedButton: {
    backgroundColor: '#32CD32',
    padding: 15,
    margin: 10,
    borderRadius: 5,
    alignItems: 'center',
  },
  previewContainer: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.9)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  fullImage: {
    width: '100%',
    height: '80%',
  },
  closePreview: {
    position: 'absolute',
    top: 40,
    right: 20,
    zIndex: 1,
    backgroundColor: '#fff',
    padding: 10,
    borderRadius: 5,
  },
  closeText: {
    color: '#000',
    fontWeight: 'bold',
    
  },
  
});

export default PhotoCapture;
