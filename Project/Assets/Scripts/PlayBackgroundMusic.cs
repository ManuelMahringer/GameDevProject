using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayBackgroundMusic : MonoBehaviour
{
    private AudioSource music1Source = null;
    // Start is called before the first frame update
    void Start()
    {
        music1Source = GameObject.Find("Music1").GetComponent<AudioSource>();
        AudioClip bgMusic1 = Resources.Load("Music/background-music") as AudioClip;

        music1Source.clip = bgMusic1;
        music1Source.volume = 0.1f;
        music1Source.Play();
}

    // Update is called once per frame
    void Update()
    {

    }
}
