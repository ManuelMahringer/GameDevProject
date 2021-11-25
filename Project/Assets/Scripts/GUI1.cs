using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GUI1 : MonoBehaviour
{
    // Start is called before the first frame update
    private void OnGUI()
    {
        if (GUI.Button(new Rect(10, 10, 100, 40), "Click me") == true)
            Debug.Log("Clicked!");
    }

}
