using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class Countdown : MonoBehaviour {
    private float currentTime = 0f;

    [SerializeField] private float startingTime; 
    [SerializeField] private TMP_Text countdownNumbers;
    [SerializeField] private World world;
    void Start() {
        GetComponent<Canvas>().enabled = true;
        currentTime = startingTime;
    }

    // Update is called once per frame
    void Update() {
        currentTime -= 1 * Time.deltaTime;
        countdownNumbers.text = currentTime.ToString("0");

        if (currentTime <= 0) {
            StopCountdown();
        }
    }

    void StopCountdown() {
        world.countdownFinished = true;
        gameObject.SetActive(false);
    }
}
