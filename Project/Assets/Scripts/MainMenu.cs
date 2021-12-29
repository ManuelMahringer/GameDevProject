using System;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using System.IO;
using System.Linq;

public class MainMenu : MonoBehaviour {
    [SerializeField]
    private TMP_Dropdown mapDropdown;
    [SerializeField]
    private TMP_Dropdown gameModeDropdown;
    private List<Map> maps;
    
    private void Start() {
        string[] mapPaths = Directory.GetDirectories(Application.persistentDataPath);
        string[] mapNames = mapPaths.Select(path => path.Split(Path.DirectorySeparatorChar).Last()).ToArray();
        maps = mapPaths.Zip(mapNames, (mapPath, mapName) => new Map {Path = mapPath, Name = mapName}).ToList();
        maps.Add(new Map {Name = "Generate", Path = ""}); // dummy option to still be able to generate the random map TODO: remove
        
        mapDropdown.ClearOptions();
        mapDropdown.AddOptions(mapNames.ToList());
        mapDropdown.AddOptions(new List<string> { "Generate" }); // dummy option to still be able to generate the random map TODO: remove
        
        gameModeDropdown.ClearOptions();
        gameModeDropdown.AddOptions(((GameMode[])Enum.GetValues(typeof(GameMode))).Select(gm => gm.ToString()).ToList());
    }

    public void PlayGame() {
        SceneManager.LoadScene(SceneManager.GetActiveScene().buildIndex + 1);
        ComponentManager.Map = maps[mapDropdown.value];
        ComponentManager.gameMode = (GameMode)gameModeDropdown.value;
        Debug.Log(ComponentManager.Map.Name);
    }
    
    public void QuitGame() {
        Application.Quit();
    }
    
    public void SetClient () {
        Debug.Log("ComponentManager set to Client");
        ComponentManager.mode = Mode.Client;
    }
    
    public void SetServer () {
        Debug.Log("ComponentManager set to Server");
        ComponentManager.mode = Mode.Server;
    }
    
    public void SetHost () {
        Debug.Log("ComponentManager set to Host");
        ComponentManager.mode = Mode.Host;
    }
}
