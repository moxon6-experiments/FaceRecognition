import numpy as np

class ROCCalculation:
    def __init__(self):
        self.collected_data = []

    def add_result(self, user, result_set):

        for result in result_set:
            genuine = (user[:5].lower() == result.name[:5].lower())

            distance = result.distance
            self.collected_data.append((distance, genuine))

    def calculate(self):

        """
        FAR = impostor_positive / (impostor_positive + impostor_negative)
        FRR = genuine_negative / (genuine_positive + genuine_negative)
        """

        error_points = []

        for threshold in np.arange(80, 100, 0.1):
            genuine_positive = 0
            impostor_positive = 0
            genuine_negative = 0
            impostor_negative = 0
            for distance, genuine in self.collected_data:

                if distance < threshold:
                    positive = True
                else:
                    positive = False

                if genuine:
                    if positive:
                        genuine_positive += 1

                    else:
                        genuine_negative += 1
                else:
                    if positive:
                        impostor_positive += 1

                    else:
                        impostor_negative += 1

            if impostor_positive + impostor_positive > 0:
                far = impostor_positive / (impostor_positive + impostor_negative)
            else:
                continue
            if genuine_positive + genuine_negative > 0:
                frr = genuine_negative / (genuine_positive + genuine_negative)
            else:
                continue

            if genuine_positive + impostor_negative + genuine_negative + impostor_positive > 0:
                accuracy = (genuine_positive + impostor_negative)/(genuine_positive + impostor_negative
                                                                   + genuine_negative + impostor_positive)
            else:
                continue

            error_points.append((threshold, far, frr, accuracy))

        return error_points

    def save_data(self):
        error_points = self.calculate()
        error_points = np.array(error_points)
        np.save("error_points", error_points)
        raise SystemExit