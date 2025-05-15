import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, TextInput } from 'react-native';
import LottieView from 'lottie-react-native';
import { useNavigation } from '@react-navigation/native';

const HomeScreen = () => {
  const navigation = useNavigation();
  const [height, setHeight] = useState('');
  const [inseamLength, setInseamLength] = useState('');
  const [armLength, setArmLength] = useState('');

  const navigateToRecord = () => {
    navigation.navigate('Record', { height, inseamLength, armLength });
  };

  const navigateToUpload = () => {
    navigation.navigate('Upload', { height, inseamLength, armLength });
  };

  return (
    <View style={styles.container}>
      <LottieView
        source={require('../assets/animation.json')}
        autoPlay
        loop
        style={styles.animation}
      />

      <TextInput
        style={styles.input}
        placeholder="Height (in cm)"
        placeholderTextColor={'#7A777A'}
        keyboardType="numeric"
        value={height}
        onChangeText={setHeight}
      />
      <TextInput
        style={styles.input}
        placeholder="Inseam Length (in cm)"
        placeholderTextColor={'#7A777A'}
        keyboardType="numeric"
        value={inseamLength}
        onChangeText={setInseamLength}
      />
      <TextInput
        style={styles.input}
        placeholder="Arm Length (in cm)"
        placeholderTextColor={'#7A777A'}
        keyboardType="numeric"
        value={armLength}
        onChangeText={setArmLength}
      />

      <TouchableOpacity
        style={styles.button}
        onPress={navigateToRecord}
      >
        <Text style={styles.buttonText}>Record</Text>
      </TouchableOpacity>
      <TouchableOpacity
        style={styles.button}
        onPress={navigateToUpload}
      >
        <Text style={styles.buttonText}>Upload</Text>
      </TouchableOpacity>
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
  animation: {
    width: 300,
    height: 300,
  },
  input: {
    height: 40,
    borderColor: '#6200ee',
    borderWidth: 1,
    borderRadius: 5,
    paddingHorizontal: 10,
    marginVertical: 10,
    width: '80%',
    color: 'black'
  },
  button: {
    backgroundColor: '#6200ee',
    padding: 15,
    borderRadius: 10,
    marginVertical: 10,
    width: '80%',
    alignItems: 'center',
  },
  buttonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
});

export default HomeScreen;