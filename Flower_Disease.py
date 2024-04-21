import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from PIL import Image

class Preprocessor:
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size

    def preprocess_image(self, image_path):
        try:
            image = Image.open(image_path)
            image = image.resize(self.image_size)
            return np.array(image)
        except Exception as e:
            print(f"Error processing image: {e}")
            return None

class FeatureExtractor:
    def __init__(self):
        pass

    def extract_features(self, image):
        # Example: Using mean and standard deviation of pixel intensities as features
        mean_intensity = np.mean(image)
        std_intensity = np.std(image)
        return mean_intensity, std_intensity

class FuzzyLeafDiseaseClassifier:
    def __init__(self):
        # Define fuzzy variables and membership functions
        self.lesion_size = ctrl.Antecedent(np.arange(0, 256, 1), 'lesion_size')
        self.lesion_density = ctrl.Antecedent(np.arange(0, 256, 1), 'lesion_density')
        self.disease = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'degree_of_disease')

        # Define membership functions for lesion size
        self.lesion_size['small'] = fuzz.trimf(self.lesion_size.universe, [0, 100, 200])
        self.lesion_size['medium'] = fuzz.trimf(self.lesion_size.universe, [0, 50, 100])
        self.lesion_size['large'] = fuzz.trimf(self.lesion_size.universe, [50, 150, 255])

        # Define membership functions for lesion density
        self.lesion_density['low'] = fuzz.trimf(self.lesion_density.universe, [0, 50, 100])
        self.lesion_density['medium'] = fuzz.trimf(self.lesion_density.universe, [50, 100, 150])
        self.lesion_density['high'] = fuzz.trimf(self.lesion_density.universe, [100, 150, 255])

        # Define membership functions for disease
        self.disease['healthy'] = fuzz.trimf(self.disease.universe, [0, 0, 0.5])
        self.disease['diseased'] = fuzz.trimf(self.disease.universe, [0.5, 1, 1])

        # Define fuzzy rules
        self.rule1 = ctrl.Rule(self.lesion_size['small'] & self.lesion_density['low'], self.disease['healthy'])
        self.rule2 = ctrl.Rule(self.lesion_size['large'] | self.lesion_density['high'], self.disease['diseased'])

        # Define fuzzy system
        self.system = ctrl.ControlSystem([self.rule1, self.rule2])
        self.simulator = ctrl.ControlSystemSimulation(self.system)

    def classify(self, lesion_size, lesion_density):
        try:
            self.simulator.input['lesion_size'] = lesion_size
            self.simulator.input['lesion_density'] = lesion_density
            self.simulator.compute()
            degree_of_membership = self.simulator.output['degree_of_disease']
            print(f"Degree of Membership: {degree_of_membership}")
            if degree_of_membership > 0.5:
                return "Healthy"
            else:
                return "Diseased"
        except Exception as e:
            print(f"Error classifying leaf: {e}")
            return None

if __name__ == "__main__":
    # Load and preprocess a single image
    image_path = r"C:\Users\Satoshi\OneDrive\Desktop\Data\PERSONAL_GROWTH\mini-projects\Images\Leaf_image_detection\potato class\potato\Potato___Late_blight\5d392db5-bf54-41f4-b76b-e3935dfe0154___RS_LB 3133.JPG"
    preprocessor = Preprocessor()
    feature_extractor = FeatureExtractor()
    classifier = FuzzyLeafDiseaseClassifier()

    image = preprocessor.preprocess_image(image_path)
    if image is not None:
        features = feature_extractor.extract_features(image)
        if features is not None:
            result = classifier.classify(features[0], features[1])
            if result is not None:
                print(f"Prediction: {result}")
