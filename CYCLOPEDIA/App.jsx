import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import HomeScreen from './components/HomeScreen';
import Record from './components/Record';
import Upload from './components/Upload';
import Photos from './components/Photos';
import { SafeAreaProvider } from 'react-native-safe-area-context'; // Import SafeAreaProvider
import LoadingScreen from './components/LoadingScreen';
import ResultScreen from './components/ResultScreen';

const Stack = createStackNavigator();

const App = () => {
  return (
    
      <NavigationContainer>
        <Stack.Navigator initialRouteName="Home">
          <Stack.Screen name="Home" component={HomeScreen} />
          <Stack.Screen name="Record" component={Record} options={{ headerShown: false }}/>
          <Stack.Screen name="Upload" component={Upload} />
          <Stack.Screen name="Photos" component={Photos} />
          <Stack.Screen name="Loading" component={LoadingScreen} options={{ headerShown: false }} />
        <Stack.Screen name="Result" component={ResultScreen} options={{ headerShown: false }} />
        </Stack.Navigator>
      </NavigationContainer>
   
  );
};

export default App;