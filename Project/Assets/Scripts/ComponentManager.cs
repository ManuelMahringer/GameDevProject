using System;
using UnityEngine;
using System.Collections.Generic;
using Unity.Netcode;

public class ComponentManager : MonoBehaviour 
{
    public static ComponentManager control;      // cheeky self-reference
    public static Mode mode;                    // our component reference
    public static Map Map { get; set; }

    void Awake()
    {
        control = this;                          // linking the self-reference
        DontDestroyOnLoad(transform.gameObject); // set to dont destroy
    }
}

public enum Mode
{
    Server,
    Client,
    Host
}

public class Map {
    public string Path { get; set; }
    public string Name { get; set; }
}