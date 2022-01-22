using UnityEngine;

public class InGameMenu : MonoBehaviour {
    [SerializeField] public GameObject mainMenu;
    [SerializeField] public GameObject controlsMenu;
    private Player _player;
    
    // Update is called once per frame
    public void Show(bool active, Player player) {
        _player = player;
        mainMenu.SetActive(active);
    }

    public void Quit() {
        Application.Quit();
    }

    public void Resume() {
        Debug.Log("resume called");
        _player.ActivateMouse();
    }
}
