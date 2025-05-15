import React from 'react';
import { View, StyleSheet, TouchableOpacity, Text, FlatList } from 'react-native';
import Video from 'react-native-video';
import { useNavigation } from '@react-navigation/native';

const ResultScreen = ({ route }) => {
  const { processedVideoUri, predictions } = route.params; // Get predictions from route.params
  const navigation = useNavigation();

  // Function to render the predictions table
  const renderTable = () => {
    if (!predictions) return null;

    // Transform predictions into an array of objects
    const tableData = Object.keys(predictions).map((key) => ({
      name: key,
      prediction: predictions[key] === 1 ? '✅' : '❌',
    }));

    return (
      <View style={styles.tableContainer}>
        <Text style={styles.tableHeader}>Predictions</Text>
        <View style={styles.headerRow}>
          <Text style={styles.headerCell}>Dysfunction</Text>
          <Text style={styles.headerCell}>Prediction</Text>
        </View>
        <FlatList
          data={tableData}
          keyExtractor={(item) => item.name}
          renderItem={({ item }) => (
            <View style={styles.row}>
              <Text style={styles.cell}>{item.name}</Text>
              <Text style={styles.cell}>{item.prediction}</Text>
            </View>
          )}
        />
      </View>
    );
  };

  return (
    <View style={styles.container}>
      {/* Back Button */}
      <TouchableOpacity onPress={() => navigation.navigate('Home')} style={styles.backButton}>
        <Text style={styles.backButtonText}>Back</Text>
      </TouchableOpacity>

      {/* Video Player */}
      <Video
        source={{ uri: processedVideoUri }}
        style={styles.video}
        controls
        resizeMode="contain"
      />

      {/* Display the predictions table */}
      {renderTable()}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f5f5f5',
    padding: 16,
  },
  video: {
    width: '100%',
    height: 300,
    marginBottom: 20,
  },
  backButton: {
    position: 'absolute',
    top: 40,
    left: 20,
    backgroundColor: '#6200ee',
    padding: 10,
    borderRadius: 5,
  },
  backButtonText: {
    color: 'white',
    fontWeight: 'bold',
  },
  tableContainer: {
    width: '100%',
    marginTop: 20,
  },
  tableHeader: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 10,
    textAlign: 'center',
  },
  headerRow: {
    flexDirection: 'row',
    borderBottomWidth: 1,
    borderBottomColor: '#000',
    paddingBottom: 8,
    marginBottom: 8,
  },
  headerCell: {
    flex: 1,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  row: {
    flexDirection: 'row',
    marginBottom: 8,
  },
  cell: {
    flex: 1,
    textAlign: 'center',
  },
});

export default ResultScreen;