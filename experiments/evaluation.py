import torch


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():  # gradient를 계산하지 않고
        maxk = max(topk)  # topk 튜플 중 최댓값
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)  # 출력값 중 상위 k개의 값(예측-여기서는 제외)과 인덱스 추출
        pred = pred.t()  # pred tensor를 transpose
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # 모델의 예측결과(pred)를 실제 정답(target)과 비교
        # boolean으로 출력됨 -> True / False 형태로

        res = []  # acc를 저장할 빈 리스트
        for k in topk:  # loop - 각 k값에 대해 정확도 계산
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            # 상위 k개의 예측에 대해서만 정답 여부를 선택. (k, batch_size)
            # .sum(0, keepdim=True) : 1D에서 값을 모두 더해, 정답 개수를 계산
            res.append(correct_k.mul_(100.0 / batch_size))
            # .mul_(100 / batch_size) : (맞춘 개수/배치 크기) x 100 = 정확도(%) 계산

        return res


