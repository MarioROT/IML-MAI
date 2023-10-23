{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMPGRUshG6HMB+79Fpqirq4"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Tjkg7Vz0QjN6"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "\n",
        "class FCM:\n",
        "  #Define all parameter of the algorithm and give a predefined value to the most common used\n",
        "    def __init__(self, C, max_iters=100, m=2, eps=0.001):\n",
        "        self.C = C\n",
        "        self.max_iters = max_iters\n",
        "        self.m = m\n",
        "        self.eps = eps\n",
        "\n",
        "  #define the main process\n",
        "    def fit(self, data):\n",
        "        self.data = data\n",
        "        self.instances, self.features = data.shape\n",
        "        #Create the random U matrix the first time that the algorithm runs\n",
        "        self.memberships = np.random.rand(self.instances, self.C)\n",
        "        self.centers = np.random.rand(self.C, self.features)\n",
        "\n",
        "        for _ in range(self.max_iters):\n",
        "            prev_centers = np.copy(self.centers)\n",
        "\n",
        "            # Update cluster centers\n",
        "            self.update_centers()\n",
        "\n",
        "            # Calculate memberships\n",
        "            self.update_memberships()\n",
        "\n",
        "            # Calculate the change in the iteration of new cluster centers\n",
        "            change = np.sum(np.abs(self.centers - prev_centers))\n",
        "\n",
        "            if change < self.eps:\n",
        "                break\n",
        "\n",
        "    def update_centers(self):\n",
        "      #With the membership matrix we calculate the centers\n",
        "        for center in range(self.C):\n",
        "            numerator = np.sum((self.memberships[:, center] ** self.m).reshape(-1, 1) * self.data, axis=0)\n",
        "            denominator = np.sum(self.memberships[:, center] ** self.m)\n",
        "            self.centers[center, :] = numerator / denominator\n",
        "\n",
        "    def update_memberships(self):\n",
        "      #given the new centers we can calculate the membership matrix\n",
        "        distances = np.zeros((self.instances, self.C))\n",
        "        for value in range(self.C):\n",
        "            distances[:, value] = np.linalg.norm(self.data - self.centers[value, :], axis=1)\n",
        "\n",
        "        for i in range(self.instances):\n",
        "            for j in range(self.C):\n",
        "                membership_sum = np.sum((distances[i, j] / distances[i, :]) ** (2 / (self.m - 1)))\n",
        "                self.memberships[i, j] = 1 / membership_sum\n",
        "\n",
        "    def predict(self):\n",
        "        return np.argmax(self.memberships, axis=1)\n",
        "\n"
      ]
    }
  ]
}