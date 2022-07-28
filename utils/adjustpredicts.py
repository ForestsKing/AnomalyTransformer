def adjust_predicts(label, pred):
    anomaly_state = False

    for i in range(len(label)):
        if label[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if label[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(label)):
                if label[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1

        elif label[i] == 0:
            anomaly_state = False

        if anomaly_state:
            pred[i] = 1

    return pred
