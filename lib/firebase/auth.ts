import {
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signInWithPopup,
  GoogleAuthProvider,
  signOut,
  sendPasswordResetEmail,
  updateProfile,
  User,
  UserCredential,
} from "firebase/auth";
import { doc, setDoc, getDoc, serverTimestamp } from "firebase/firestore";
import { auth, db } from "./config";

export interface UserData {
  uid: string;
  email: string;
  displayName: string | null;
  photoURL: string | null;
  createdAt: any;
  updatedAt: any;
  provider?: string;
}

// Google Auth Provider
const googleProvider = new GoogleAuthProvider();
googleProvider.setCustomParameters({
  prompt: "select_account",
});

/**
 * Sign up with email and password
 */
export async function signUpWithEmail(
  email: string,
  password: string,
  displayName?: string
): Promise<UserCredential> {
  try {
    const userCredential = await createUserWithEmailAndPassword(
      auth,
      email,
      password
    );

    // Update display name if provided
    if (displayName && userCredential.user) {
      await updateProfile(userCredential.user, {
        displayName: displayName,
      });
    }

    // Save user data to Firestore
    await saveUserToFirestore(userCredential.user, "email");

    return userCredential;
  } catch (error: any) {
    throw error;
  }
}

/**
 * Sign in with email and password
 */
export async function signInWithEmail(
  email: string,
  password: string
): Promise<UserCredential> {
  try {
    const userCredential = await signInWithEmailAndPassword(
      auth,
      email,
      password
    );

    // Update last login time
    await updateUserLastLogin(userCredential.user.uid);

    return userCredential;
  } catch (error: any) {
    throw error;
  }
}

/**
 * Sign in with Google
 */
export async function signInWithGoogle(): Promise<UserCredential> {
  try {
    const userCredential = await signInWithPopup(auth, googleProvider);

    // Save user data to Firestore
    await saveUserToFirestore(userCredential.user, "google");

    return userCredential;
  } catch (error: any) {
    throw error;
  }
}

/**
 * Sign out
 */
export async function signOutUser(): Promise<void> {
  try {
    await signOut(auth);
  } catch (error: any) {
    throw error;
  }
}

/**
 * Send password reset email
 */
export async function resetPassword(email: string): Promise<void> {
  try {
    await sendPasswordResetEmail(auth, email);
  } catch (error: any) {
    throw error;
  }
}

/**
 * Save user data to Firestore
 */
async function saveUserToFirestore(
  user: User,
  provider: string = "email"
): Promise<void> {
  try {
    const userRef = doc(db, "users", user.uid);
    const userSnap = await getDoc(userRef);

    const userData: Partial<UserData> = {
      uid: user.uid,
      email: user.email || "",
      displayName: user.displayName || null,
      photoURL: user.photoURL || null,
      provider: provider,
      updatedAt: serverTimestamp(),
    };

    // Only set createdAt if user doesn't exist
    if (!userSnap.exists()) {
      userData.createdAt = serverTimestamp();
    }

    await setDoc(userRef, userData, { merge: true });
  } catch (error: any) {
    // Soft-fail: Auth is the source of truth for sign-in; profile write
    // must not fail Google/email authentication.
    console.error("Error saving user to Firestore:", error);
  }
}

/**
 * Update user's last login time
 */
async function updateUserLastLogin(uid: string): Promise<void> {
  try {
    const userRef = doc(db, "users", uid);
    await setDoc(
      userRef,
      {
        lastLoginAt: serverTimestamp(),
        updatedAt: serverTimestamp(),
      },
      { merge: true }
    );
  } catch (error: any) {
    console.error("Error updating last login:", error);
  }
}

/**
 * Get user data from Firestore
 */
export async function getUserData(uid: string): Promise<UserData | null> {
  try {
    const userRef = doc(db, "users", uid);
    const userSnap = await getDoc(userRef);

    if (userSnap.exists()) {
      return userSnap.data() as UserData;
    }
    return null;
  } catch (error: any) {
    console.error("Error getting user data:", error);
    return null;
  }
}
