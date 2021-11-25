using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;
using System;
using TMPro;

public class IncreasePoints : MonoBehaviour
{
    private UnityAction onAddPoints;
    private TextMeshProUGUI tmpObject;

    // Start is called before the first frame update
    private void Awake()
    {
        Debug.Log("Awaking");
        onAddPoints = new UnityAction(OnAddPoints);
        tmpObject = GetComponent<TextMeshProUGUI>();
        tmpObject.text = "0";
    }

    private void OnEnable()
    {
        EventManager.StartListening("AddPoints", onAddPoints);
    }

    private void OnDisable()
    {
        EventManager.StopListening("AddPoints", onAddPoints);

    }
     

    // Update is called once per frame
    private void OnAddPoints()
    {
        Debug.Log("AddPoints received");

        tmpObject.text = Convert.ToString(Int32.Parse(tmpObject.text) + 5);
    }
} 