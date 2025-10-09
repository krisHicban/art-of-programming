# Complete OOP Project: Course Management System

class Person:
    """Base class for all people in the system"""
    
    def __init__(self, name, email, age):
        self.name = name
        self.email = email
        self.age = age
        self.id = self._generate_id()
    
    def _generate_id(self):
        """Generate a unique ID"""
        import random
        return f"ID{random.randint(1000, 9999)}"
    
    def get_info(self):
        """Get basic person information"""
        return f"{self.name} ({self.email}) - Age: {self.age}"

class Student(Person):
    """Student class inheriting from Person"""
    
    def __init__(self, name, email, age):
        super().__init__(name, email, age)
        self.enrolled_courses = []
        self.grades = {}
    
    def enroll_course(self, course):
        """Enroll in a course"""
        if course not in self.enrolled_courses:
            self.enrolled_courses.append(course)
            self.grades[course.course_id] = []
            return f"{self.name} enrolled in {course.title}"
        return f"{self.name} is already enrolled in {course.title}"
    
    def add_grade(self, course_id, grade):
        """Add a grade for a course"""
        if course_id in self.grades:
            if 0 <= grade <= 100:
                self.grades[course_id].append(grade)
                return f"Grade {grade} added for course {course_id}"
            return "Grade must be between 0 and 100"
        return "Student not enrolled in this course"
    
    def get_average_grade(self, course_id):
        """Calculate average grade for a course"""
        if course_id in self.grades and self.grades[course_id]:
            return sum(self.grades[course_id]) / len(self.grades[course_id])
        return 0
    
    def get_transcript(self):
        """Get student transcript"""
        transcript = f"\n=== Transcript for {self.name} ===\n"
        for course in self.enrolled_courses:
            avg_grade = self.get_average_grade(course.course_id)
            transcript += f"Course: {course.title} - Average: {avg_grade:.1f}\n"
        return transcript

class Trainer(Person):
    """Trainer class inheriting from Person"""
    
    def __init__(self, name, email, age, specialization):
        super().__init__(name, email, age)
        self.specialization = specialization
        self.courses_taught = []
        self.salary = 0
    
    def assign_course(self, course):
        """Assign a course to the trainer"""
        if course not in self.courses_taught:
            self.courses_taught.append(course)
            course.trainer = self
            return f"{self.name} assigned to teach {course.title}"
        return f"{self.name} is already teaching {course.title}"
    
    def grade_student(self, student, course_id, grade):
        """Grade a student in a course"""
        # Check if trainer teaches this course
        teaching_course = any(course.course_id == course_id for course in self.courses_taught)
        if teaching_course:
            return student.add_grade(course_id, grade)
        return "Trainer is not teaching this course"
    
    def get_teaching_load(self):
        """Get trainer's teaching load"""
        if not self.courses_taught:
            return f"{self.name} is not teaching any courses"
        
        load_info = f"\n=== {self.name}'s Teaching Load ===\n"
        load_info += f"Specialization: {self.specialization}\n"
        load_info += "Courses:\n"
        for course in self.courses_taught:
            load_info += f"  - {course.title} ({len(course.students)} students)\n"
        return load_info

class Course:
    """Course class"""
    
    def __init__(self, title, course_code, duration_weeks):
        self.title = title
        self.course_code = course_code
        self.course_id = f"{course_code}_{self._generate_course_id()}"
        self.duration_weeks = duration_weeks
        self.students = []
        self.trainer = None
        self.is_active = True
    
    def _generate_course_id(self):
        """Generate unique course ID"""
        import random
        return random.randint(100, 999)
    
    def add_student(self, student):
        """Add student to course"""
        if student not in self.students and self.is_active:
            self.students.append(student)
            return student.enroll_course(self)
        return "Cannot add student to course"
    
    def remove_student(self, student):
        """Remove student from course"""
        if student in self.students:
            self.students.remove(student)
            if self.course_id in student.grades:
                del student.grades[self.course_id]
            return f"{student.name} removed from {self.title}"
        return "Student not in course"
    
    def get_course_info(self):
        """Get course information"""
        trainer_name = self.trainer.name if self.trainer else "No trainer assigned"
        status = "Active" if self.is_active else "Inactive"
        
        info = f"\n=== Course: {self.title} ===\n"
        info += f"Code: {self.course_code} | ID: {self.course_id}\n"
        info += f"Duration: {self.duration_weeks} weeks\n"
        info += f"Trainer: {trainer_name}\n"
        info += f"Students: {len(self.students)}\n"
        info += f"Status: {status}\n"
        return info
    
    def get_class_average(self):
        """Calculate class average"""
        if not self.students:
            return 0
        
        total_average = 0
        students_with_grades = 0
        
        for student in self.students:
            avg = student.get_average_grade(self.course_id)
            if avg > 0:
                total_average += avg
                students_with_grades += 1
        
        return total_average / students_with_grades if students_with_grades > 0 else 0

class CourseManagementSystem:
    """Main system to manage courses, students, and trainers"""
    
    def __init__(self):
        self.courses = []
        self.students = []
        self.trainers = []
    
    def add_course(self, title, course_code, duration_weeks):
        """Add a new course"""
        course = Course(title, course_code, duration_weeks)
        self.courses.append(course)
        return f"Course {title} created with ID: {course.course_id}"
    
    def add_student(self, name, email, age):
        """Add a new student"""
        student = Student(name, email, age)
        self.students.append(student)
        return f"Student {name} added with ID: {student.id}"
    
    def add_trainer(self, name, email, age, specialization):
        """Add a new trainer"""
        trainer = Trainer(name, email, age, specialization)
        self.trainers.append(trainer)
        return f"Trainer {name} added with ID: {trainer.id}"
    
    def enroll_student_in_course(self, student_id, course_id):
        """Enroll a student in a course"""
        student = self.find_student(student_id)
        course = self.find_course(course_id)
        
        if student and course:
            return course.add_student(student)
        return "Student or course not found"
    
    def assign_trainer_to_course(self, trainer_id, course_id):
        """Assign trainer to course"""
        trainer = self.find_trainer(trainer_id)
        course = self.find_course(course_id)
        
        if trainer and course:
            return trainer.assign_course(course)
        return "Trainer or course not found"
    
    def find_student(self, student_id):
        """Find student by ID"""
        return next((s for s in self.students if s.id == student_id), None)
    
    def find_trainer(self, trainer_id):
        """Find trainer by ID"""
        return next((t for t in self.trainers if t.id == trainer_id), None)
    
    def find_course(self, course_id):
        """Find course by ID"""
        return next((c for c in self.courses if c.course_id == course_id), None)
    
    def generate_system_report(self):
        """Generate comprehensive system report"""
        report = "\n" + "="*50
        report += "\n        COURSE MANAGEMENT SYSTEM REPORT"
        report += "\n" + "="*50
        
        report += f"\nTotal Courses: {len(self.courses)}"
        report += f"\nTotal Students: {len(self.students)}"
        report += f"\nTotal Trainers: {len(self.trainers)}"
        
        report += "\n\nCOURSE DETAILS:"
        for course in self.courses:
            report += course.get_course_info()
            if course.students:
                class_avg = course.get_class_average()
                report += f"Class Average: {class_avg:.1f}\n"
        
        return report

# Demonstration of the Course Management System
print("=== Course Management System Demo ===")

# Create the system
cms = CourseManagementSystem()

# Add courses
print(cms.add_course("Python Programming", "PY101", 12))
print(cms.add_course("Data Science Fundamentals", "DS201", 16))
print(cms.add_course("Web Development", "WD301", 14))

# Add trainers
print(cms.add_trainer("Dr. Alice Smith", "alice@email.com", 35, "Python Programming"))
print(cms.add_trainer("Prof. Bob Johnson", "bob@email.com", 42, "Data Science"))

# Add students
print(cms.add_student("Emma Wilson", "emma@email.com", 22))
print(cms.add_student("James Brown", "james@email.com", 24))
print(cms.add_student("Sofia Garcia", "sofia@email.com", 21))

# Get IDs for demonstration (in real system, you'd store these)
python_course = cms.courses[0]
trainer_alice = cms.trainers[0]
student_emma = cms.students[0]
student_james = cms.students[1]

# Assign trainer to course
print(cms.assign_trainer_to_course(trainer_alice.id, python_course.course_id))

# Enroll students in courses
print(cms.enroll_student_in_course(student_emma.id, python_course.course_id))
print(cms.enroll_student_in_course(student_james.id, python_course.course_id))

# Add some grades
trainer_alice.grade_student(student_emma, python_course.course_id, 95)
trainer_alice.grade_student(student_emma, python_course.course_id, 88)
trainer_alice.grade_student(student_james, python_course.course_id, 92)
trainer_alice.grade_student(student_james, python_course.course_id, 85)

# Generate reports
print(student_emma.get_transcript())
print(trainer_alice.get_teaching_load())
print(cms.generate_system_report())