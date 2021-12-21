using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class MainMenu : MonoBehaviour
{
    public void PlayGame() {
        SceneManager.LoadScene(SceneManager.GetActiveScene().buildIndex + 1);
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
